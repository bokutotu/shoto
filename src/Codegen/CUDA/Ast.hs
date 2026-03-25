{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.CUDA.Ast (
    CudaAstError (..),
    CudaExpr (..),
    CudaIndexBinding (..),
    CudaStmt (..),
    CudaProgram (..),
    renderCudaIndexLinearExpr,
    lowerToCudaProgram,
) where

import           Codegen.GenIR    (GenExpr (..), GenProgram (..), GenStmt (..))
import qualified Data.Map.Strict  as Map
import qualified Data.Set         as Set
import           FrontendIR.Types (IterName, ParamName, TensorName,
                                   tensorNameToString)
import           IR.Name          (KernelName (..))
import           IR.Tensor        (TensorDecl (..), TensorRef (..))

data CudaAstError
    = ErrCudaAstReductionNotSupported
    | ErrCudaAstRankTooLarge Int
    | ErrCudaAstTensorShapeMismatch String Int Int
    | ErrCudaAstMalformedGenProgram
    deriving (Eq, Show)

data CudaDim
    = CudaX
    | CudaY
    | CudaZ
    deriving (Eq, Show)

data CudaExpr
    = CudaVar IterName
    | CudaExtentVar ParamName
    | CudaInt Int
    | CudaLoad TensorName [CudaExpr]
    | CudaAdd CudaExpr CudaExpr
    | CudaMul CudaExpr CudaExpr
    deriving (Eq, Show)

data CudaIndexBinding = CudaIndexBinding
    { cudaIndexVar :: IterName
    , cudaIndexDim :: CudaDim
    }
    deriving (Eq, Show)

data CudaStmt
    = CudaIfAllLessThan
        { cudaIfBindings :: [(IterName, ParamName)]
        , cudaIfBody :: [CudaStmt]
        }
    | CudaAssign
        { cudaAssignTarget :: TensorRef CudaExpr
        , cudaAssignExpr :: CudaExpr
        }
    deriving (Eq, Show)

data CudaProgram = CudaProgram
    { cudaKernelName :: KernelName
    , cudaExtentParams :: [ParamName]
    , cudaTensorArgs :: [TensorName]
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
            { cudaKernelName = KernelName "shoto_kernel_cuda"
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
    Map.Map TensorName [ParamName] ->
    [ParamName] ->
    GenStmt ->
    Either CudaAstError (TensorRef CudaExpr, CudaExpr, [TensorName], [IterName])
lowerKernelStmt tensorShapes extentParams stmt =
    case stmt of
        GenAssign{} -> do
            if length stmt.genTarget.tensorIndices /= length extentParams
                then Left ErrCudaAstMalformedGenProgram
                else do
                    storeRef <- lowerTensorRef tensorShapes stmt.genTarget
                    rhsExpr <- lowerExpr tensorShapes stmt.genExpr
                    let tensorArgs =
                            uniqueStable
                                ( stmt.genTarget.tensorName
                                    : collectLoadTensors stmt.genExpr
                                )
                    pure (storeRef, rhsExpr, tensorArgs, stmt.genTarget.tensorIndices)
        GenReduction{} -> Left ErrCudaAstReductionNotSupported
        GenFor{} -> Left ErrCudaAstMalformedGenProgram

collectLoadTensors :: GenExpr -> [TensorName]
collectLoadTensors expr =
    case expr of
        GenConst _ -> []
        GenLoad ref -> [ref.tensorName]
        GenAdd lhs rhs -> collectLoadTensors lhs <> collectLoadTensors rhs
        GenMul lhs rhs -> collectLoadTensors lhs <> collectLoadTensors rhs

lowerExpr ::
    Map.Map TensorName [ParamName] ->
    GenExpr ->
    Either CudaAstError CudaExpr
lowerExpr tensorShapes expr =
    case expr of
        GenConst value -> pure $ CudaInt value
        GenLoad ref -> lowerTensorLoad tensorShapes ref
        GenAdd lhs rhs -> CudaAdd <$> lowerExpr tensorShapes lhs <*> lowerExpr tensorShapes rhs
        GenMul lhs rhs -> CudaMul <$> lowerExpr tensorShapes lhs <*> lowerExpr tensorShapes rhs

lowerTensorLoad ::
    Map.Map TensorName [ParamName] ->
    TensorRef IterName ->
    Either CudaAstError CudaExpr
lowerTensorLoad tensorShapes ref = do
    tensorRef <- lowerTensorRef tensorShapes ref
    pure $ CudaLoad tensorRef.tensorName tensorRef.tensorIndices

lowerTensorRef ::
    Map.Map TensorName [ParamName] ->
    TensorRef IterName ->
    Either CudaAstError (TensorRef CudaExpr)
lowerTensorRef tensorShapes ref = do
    shapeParams <- expectTensorShape tensorShapes ref.tensorName
    linearIndex <-
        linearizeRowMajor
            (tensorNameToString ref.tensorName)
            shapeParams
            (CudaVar <$> ref.tensorIndices)
    pure
        TensorRef
            { tensorName = ref.tensorName
            , tensorIndices = [linearIndex]
            }

linearizeRowMajor ::
    String ->
    [ParamName] ->
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
    Map.Map TensorName [ParamName] ->
    TensorName ->
    Either CudaAstError [ParamName]
expectTensorShape tensorShapes tensor =
    case Map.lookup tensor tensorShapes of
        Just shape -> pure shape
        Nothing ->
            Left $
                ErrCudaAstTensorShapeMismatch
                    (tensorNameToString tensor)
                    0
                    0

buildTensorShapeMap :: [TensorDecl] -> Map.Map TensorName [ParamName]
buildTensorShapeMap tensorDecls =
    Map.fromList
        [ (tensorDecl.tensor, tensorDecl.shape)
        | tensorDecl <- tensorDecls
        ]

buildIndexBindings :: [IterName] -> [CudaIndexBinding]
buildIndexBindings logicalIndices =
    [ CudaIndexBinding
        { cudaIndexVar = iterName
        , cudaIndexDim = dim
        }
    | (iterName, dim) <- zip (reverse logicalIndices) [CudaX, CudaY, CudaZ]
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
