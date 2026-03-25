{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.CUDA.Ast (
    CudaAstError (..),
    CudaKernelName (..),
    CudaTensorName (..),
    CudaExpr (..),
    CudaTensorRef (..),
    CudaIndexBinding (..),
    CudaStmt (..),
    CudaProgram (..),
    renderCudaIndexLinearExpr,
    lowerToCudaProgram,
) where

import           Codegen.GenIR    (AstIterVar, ExtentParamName, GenExpr (..),
                                   GenProgram (..), GenStmt (..),
                                   GenTensorDecl (..), GenTensorRef (..))
import qualified Data.Map.Strict  as Map
import qualified Data.Set         as Set
import           Data.String      (IsString (fromString))
import           FrontendIR.Types (TensorName, tensorNameToString)

data CudaAstError
    = ErrCudaAstReductionNotSupported
    | ErrCudaAstRankTooLarge Int
    | ErrCudaAstTensorShapeMismatch String Int Int
    | ErrCudaAstMalformedGenProgram
    deriving (Eq, Show)

newtype CudaKernelName = CudaKernelName String deriving (Eq, Ord, Show)

instance IsString CudaKernelName where fromString = CudaKernelName

newtype CudaTensorName = CudaTensorName String deriving (Eq, Ord, Show)

instance IsString CudaTensorName where fromString = CudaTensorName

data CudaDim
    = CudaX
    | CudaY
    | CudaZ
    deriving (Eq, Show)

data CudaExpr
    = CudaVar AstIterVar
    | CudaExtentVar ExtentParamName
    | CudaInt Int
    | CudaLoad CudaTensorName [CudaExpr]
    | CudaAdd CudaExpr CudaExpr
    | CudaMul CudaExpr CudaExpr
    deriving (Eq, Show)

data CudaTensorRef = CudaTensorRef
    { tensorName :: CudaTensorName
    , tensorIndices :: [CudaExpr]
    }
    deriving (Eq, Show)

data CudaIndexBinding = CudaIndexBinding
    { cudaIndexVar :: AstIterVar
    , cudaIndexDim :: CudaDim
    }
    deriving (Eq, Show)

data CudaStmt
    = CudaIfAllLessThan
        { cudaIfBindings :: [(AstIterVar, ExtentParamName)]
        , cudaIfBody :: [CudaStmt]
        }
    | CudaAssign
        { cudaAssignTarget :: CudaTensorRef
        , cudaAssignExpr :: CudaExpr
        }
    deriving (Eq, Show)

data CudaProgram = CudaProgram
    { cudaKernelName :: CudaKernelName
    , cudaExtentParams :: [ExtentParamName]
    , cudaTensorArgs :: [CudaTensorName]
    , cudaIndexBindings :: [CudaIndexBinding]
    , cudaBody :: [CudaStmt]
    }
    deriving (Eq, Show)

lowerToCudaProgram :: GenProgram -> Either CudaAstError CudaProgram
lowerToCudaProgram genProgram = do
    validateRank genProgram
    stmt <- extractKernelStmt genProgram.genBody
    (target, rhsExpr, tensorArgs, logicalIndices) <-
        lowerKernelStmt tensorShapes genProgram.genExtentParams stmt
    pure
        CudaProgram
            { cudaKernelName = fromString "shoto_kernel_cuda"
            , cudaExtentParams = genProgram.genExtentParams
            , cudaTensorArgs = tensorArgs
            , cudaIndexBindings = buildIndexBindings logicalIndices
            , cudaBody =
                [ CudaIfAllLessThan
                    { cudaIfBindings = zip logicalIndices genProgram.genExtentParams
                    , cudaIfBody =
                        [ CudaAssign
                            { cudaAssignTarget = target
                            , cudaAssignExpr = rhsExpr
                            }
                        ]
                    }
                ]
            }
  where
    tensorShapes = buildTensorShapeMap genProgram.genTensorDecls

extractKernelStmt :: [GenStmt] -> Either CudaAstError GenStmt
extractKernelStmt stmts =
    case stmts of
        [GenFor{genBody = body}] -> extractKernelStmt body
        [stmt] -> pure stmt
        _ -> Left ErrCudaAstMalformedGenProgram

lowerKernelStmt ::
    Map.Map TensorName [ExtentParamName] ->
    [ExtentParamName] ->
    GenStmt ->
    Either CudaAstError (CudaTensorRef, CudaExpr, [CudaTensorName], [AstIterVar])
lowerKernelStmt tensorShapes extentParams stmt =
    case stmt of
        GenAssign{} -> do
            if length stmt.genTarget.genIndices /= length extentParams
                then Left ErrCudaAstMalformedGenProgram
                else do
                    storeRef <- lowerTensorRef tensorShapes stmt.genTarget
                    rhsExpr <- lowerExpr tensorShapes stmt.genExpr
                    let outputTensorName =
                            fromString $
                                tensorNameToString stmt.genTarget.genTensor
                        tensorArgs =
                            uniqueStable
                                ( outputTensorName
                                    : collectLoadTensors stmt.genExpr
                                )
                    pure (storeRef, rhsExpr, tensorArgs, stmt.genTarget.genIndices)
        GenReduction{} -> Left ErrCudaAstReductionNotSupported
        GenFor{} -> Left ErrCudaAstMalformedGenProgram

collectLoadTensors :: GenExpr -> [CudaTensorName]
collectLoadTensors expr =
    case expr of
        GenConst _ -> []
        GenLoad ref -> [fromString $ tensorNameToString ref.genTensor]
        GenAdd lhs rhs -> collectLoadTensors lhs <> collectLoadTensors rhs
        GenMul lhs rhs -> collectLoadTensors lhs <> collectLoadTensors rhs

lowerExpr ::
    Map.Map TensorName [ExtentParamName] ->
    GenExpr ->
    Either CudaAstError CudaExpr
lowerExpr tensorShapes expr =
    case expr of
        GenConst value -> pure $ CudaInt value
        GenLoad ref -> lowerTensorLoad tensorShapes ref
        GenAdd lhs rhs -> CudaAdd <$> lowerExpr tensorShapes lhs <*> lowerExpr tensorShapes rhs
        GenMul lhs rhs -> CudaMul <$> lowerExpr tensorShapes lhs <*> lowerExpr tensorShapes rhs

lowerTensorLoad ::
    Map.Map TensorName [ExtentParamName] ->
    GenTensorRef ->
    Either CudaAstError CudaExpr
lowerTensorLoad tensorShapes ref = do
    tensorRef <- lowerTensorRef tensorShapes ref
    pure $ CudaLoad tensorRef.tensorName tensorRef.tensorIndices

lowerTensorRef ::
    Map.Map TensorName [ExtentParamName] ->
    GenTensorRef ->
    Either CudaAstError CudaTensorRef
lowerTensorRef tensorShapes ref = do
    shapeParams <- expectTensorShape tensorShapes ref.genTensor
    linearIndex <-
        linearizeRowMajor (tensorNameToString ref.genTensor) shapeParams (CudaVar <$> ref.genIndices)
    pure
        CudaTensorRef
            { tensorName = fromString $ tensorNameToString ref.genTensor
            , tensorIndices = [linearIndex]
            }

linearizeRowMajor ::
    String ->
    [ExtentParamName] ->
    [CudaExpr] ->
    Either CudaAstError CudaExpr
linearizeRowMajor tensorName shapeParams indices
    | length shapeParams /= length indices =
        Left $
            ErrCudaAstTensorShapeMismatch
                tensorName
                (length shapeParams)
                (length indices)
    | null indices = pure $ CudaInt 0
    | otherwise =
        case (indices, shapeParams) of
            (firstIndex : restIndices, _ : restShapeParams) ->
                pure $
                    foldl'
                        step
                        firstIndex
                        (zip restIndices restShapeParams)
            _ -> pure $ CudaInt 0
  where
    step acc (indexExpr, extentParam) =
        CudaAdd
            (CudaMul acc (CudaExtentVar extentParam))
            indexExpr

expectTensorShape ::
    Map.Map TensorName [ExtentParamName] ->
    TensorName ->
    Either CudaAstError [ExtentParamName]
expectTensorShape tensorShapes tensor =
    case Map.lookup tensor tensorShapes of
        Just shape -> pure shape
        Nothing ->
            Left $
                ErrCudaAstTensorShapeMismatch
                    (tensorNameToString tensor)
                    0
                    0

buildTensorShapeMap :: [GenTensorDecl] -> Map.Map TensorName [ExtentParamName]
buildTensorShapeMap tensorDecls =
    Map.fromList
        [ (tensorDecl.genTensor, tensorDecl.genShape)
        | tensorDecl <- tensorDecls
        ]

buildIndexBindings :: [AstIterVar] -> [CudaIndexBinding]
buildIndexBindings logicalIndices =
    [ CudaIndexBinding
        { cudaIndexVar = astIter
        , cudaIndexDim = dim
        }
    | (astIter, dim) <- zip (reverse logicalIndices) [CudaX, CudaY, CudaZ]
    ]

validateRank :: GenProgram -> Either CudaAstError ()
validateRank genProgram =
    if rank <= 3
        then pure ()
        else Left $ ErrCudaAstRankTooLarge rank
  where
    rank = length genProgram.genExtentParams

uniqueStable :: (Ord a) => [a] -> [a]
uniqueStable = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
        | x `Set.member` seen = go seen xs
        | otherwise = x : go (Set.insert x seen) xs

renderCudaIndexLinearExpr :: CudaIndexBinding -> String
renderCudaIndexLinearExpr CudaIndexBinding{cudaIndexDim = dim} =
    case dim of
        CudaX -> "blockIdx.x * blockDim.x + threadIdx.x"
        CudaY -> "blockIdx.y * blockDim.y + threadIdx.y"
        CudaZ -> "blockIdx.z * blockDim.z + threadIdx.z"
