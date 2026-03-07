{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.CUDA.Ast (
    CudaAstError (..),
    CudaKernelName (..),
    CudaTensorName (..),
    CudaDim (..),
    CudaExpr (..),
    CudaTensorRef (..),
    CudaIndexBinding (..),
    CudaStmt (..),
    CudaProgram (..),
    lowerToCudaProgram,
) where

import           Codegen.GenIR    (AstIterVar, ExtentParamName,
                                   FrontendIterName, GenLoop (..),
                                   GenProgram (..), GenStmtBinding (..))
import qualified Data.Map.Strict  as Map
import qualified Data.Set         as Set
import           Data.String      (IsString (fromString))
import           FrontendIR.Types (Expr (..), IxExpr (..), Stmt (..),
                                   iterNameToString, tensorNameToString)

data CudaAstError
    = ErrCudaAstReductionNotSupported
    | ErrCudaAstExpectedSingleStoreIndex Int
    | ErrCudaAstExpectedSingleLoadIndex Int
    | ErrCudaAstUnknownIterBinding FrontendIterName
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
    = CudaIfLessThan
        { cudaIfVar :: AstIterVar
        , cudaIfBound :: ExtentParamName
        , cudaIfBody :: [CudaStmt]
        }
    | CudaAssign
        { cudaAssignTarget :: CudaTensorRef
        , cudaAssignExpr :: CudaExpr
        }
    deriving (Eq, Show)

data CudaProgram = CudaProgram
    { cudaKernelName :: CudaKernelName
    , cudaExtentParam :: ExtentParamName
    , cudaTensorArgs :: [CudaTensorName]
    , cudaIndexBinding :: CudaIndexBinding
    , cudaBody :: [CudaStmt]
    }
    deriving (Eq, Show)

lowerToCudaProgram :: CudaDim -> GenProgram -> Either CudaAstError CudaProgram
lowerToCudaProgram dim genProgram = do
    (target, rhsExpr, tensorArgs) <- lowerAssign genProgram
    let loopInfo = genProgram.loop
        indexVar = loopInfo.astIterator
    pure
        CudaProgram
            { cudaKernelName = fromString "shoto_kernel_cuda"
            , cudaExtentParam = loopInfo.extentParam
            , cudaTensorArgs = tensorArgs
            , cudaIndexBinding =
                CudaIndexBinding
                    { cudaIndexVar = indexVar
                    , cudaIndexDim = dim
                    }
            , cudaBody =
                [ CudaIfLessThan
                    { cudaIfVar = indexVar
                    , cudaIfBound = loopInfo.extentParam
                    , cudaIfBody =
                        [ CudaAssign
                            { cudaAssignTarget = target
                            , cudaAssignExpr = rhsExpr
                            }
                        ]
                    }
                ]
            }

lowerAssign :: GenProgram -> Either CudaAstError (CudaTensorRef, CudaExpr, [CudaTensorName])
lowerAssign genProgram =
    case genProgram.stmtBinding.frontendStmt of
        Reduction{} -> Left ErrCudaAstReductionNotSupported
        Assign{} -> do
            let iterMap =
                    Map.singleton
                        genProgram.loop.frontendIter
                        genProgram.stmtBinding.astIndexVar
            storeIndices <- traverse (lowerIxExpr iterMap) genProgram.stmtBinding.frontendStmt.outputIndex
            case storeIndices of
                [singleStoreIndex] -> do
                    rhsExpr <- lowerExpr iterMap genProgram.stmtBinding.frontendStmt.rhs
                    let outputTensorName =
                            fromString $
                                tensorNameToString genProgram.stmtBinding.frontendStmt.outputTensor
                        tensorArgs =
                            uniqueStable
                                ( outputTensorName
                                    : collectLoadTensors genProgram.stmtBinding.frontendStmt.rhs
                                )
                    pure
                        ( CudaTensorRef
                            { tensorName = outputTensorName
                            , tensorIndices = [singleStoreIndex]
                            }
                        , rhsExpr
                        , tensorArgs
                        )
                _ -> Left $ ErrCudaAstExpectedSingleStoreIndex (length storeIndices)

lowerExpr :: Map.Map FrontendIterName AstIterVar -> Expr -> Either CudaAstError CudaExpr
lowerExpr iterMap expr =
    case expr of
        EConst value -> pure $ CudaInt value
        ELoad tensor indices -> do
            loweredIndices <- traverse (lowerIxExpr iterMap) indices
            case loweredIndices of
                [_] -> pure $ CudaLoad (fromString $ tensorNameToString tensor) loweredIndices
                _ -> Left $ ErrCudaAstExpectedSingleLoadIndex (length loweredIndices)
        EAdd lhs rhs -> CudaAdd <$> lowerExpr iterMap lhs <*> lowerExpr iterMap rhs
        EMul lhs rhs -> CudaMul <$> lowerExpr iterMap lhs <*> lowerExpr iterMap rhs

lowerIxExpr :: Map.Map FrontendIterName AstIterVar -> IxExpr -> Either CudaAstError CudaExpr
lowerIxExpr iterMap ixExpr =
    case ixExpr of
        IxVar iterName ->
            let iterKey = fromString $ iterNameToString iterName
             in case Map.lookup iterKey iterMap of
                    Just loweredName -> pure $ CudaVar loweredName
                    Nothing -> Left $ ErrCudaAstUnknownIterBinding iterKey

collectLoadTensors :: Expr -> [CudaTensorName]
collectLoadTensors expr =
    case expr of
        EConst _ -> []
        ELoad tensor _ -> [fromString $ tensorNameToString tensor]
        EAdd lhs rhs -> collectLoadTensors lhs <> collectLoadTensors rhs
        EMul lhs rhs -> collectLoadTensors lhs <> collectLoadTensors rhs

uniqueStable :: (Ord a) => [a] -> [a]
uniqueStable = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
        | x `Set.member` seen = go seen xs
        | otherwise = x : go (Set.insert x seen) xs
