{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.C.Ast (
    CAstError (..),
    CFunctionName (..),
    CTensorName (..),
    CExpr (..),
    CTensorRef (..),
    CStmt (..),
    CProgram (..),
    lowerToCProgram,
) where

import           Codegen.GenIR    (AstIterVar, ExtentParamName,
                                   FrontendIterName, GenLoop (..),
                                   GenProgram (..), GenStmtBinding (..))
import qualified Data.Map.Strict  as Map
import qualified Data.Set         as Set
import           Data.String      (IsString (fromString))
import           FrontendIR.Types (Expr (..), IxExpr (..), Stmt (..),
                                   iterNameToString, tensorNameToString)

data CAstError
    = ErrCAstReductionNotSupported
    | ErrCAstExpectedSingleStoreIndex Int
    | ErrCAstExpectedSingleLoadIndex Int
    | ErrCAstUnknownIterBinding FrontendIterName
    deriving (Eq, Show)

newtype CFunctionName = CFunctionName String deriving (Eq, Ord, Show)

instance IsString CFunctionName where fromString = CFunctionName

newtype CTensorName = CTensorName String deriving (Eq, Ord, Show)

instance IsString CTensorName where fromString = CTensorName

data CExpr
    = CVar AstIterVar
    | CInt Int
    | CLoad CTensorName [CExpr]
    | CAdd CExpr CExpr
    | CMul CExpr CExpr
    deriving (Eq, Show)

data CTensorRef = CTensorRef
    { tensorName :: CTensorName
    , tensorIndices :: [CExpr]
    }
    deriving (Eq, Show)

data CStmt
    = CForLoop
        { cForVar :: AstIterVar
        , cForUpperBound :: ExtentParamName
        , cForBody :: [CStmt]
        }
    | CAssign
        { cAssignTarget :: CTensorRef
        , cAssignExpr :: CExpr
        }
    deriving (Eq, Show)

data CProgram = CProgram
    { cFunctionName :: CFunctionName
    , cExtentParam :: ExtentParamName
    , cTensorArgs :: [CTensorName]
    , cBody :: [CStmt]
    }
    deriving (Eq, Show)

lowerToCProgram :: GenProgram -> Either CAstError CProgram
lowerToCProgram genProgram = do
    (target, rhsExpr, tensorArgs) <- lowerAssign genProgram
    let loopInfo = genProgram.loop
    pure
        CProgram
            { cFunctionName = fromString "shoto_kernel"
            , cExtentParam = loopInfo.extentParam
            , cTensorArgs = tensorArgs
            , cBody =
                [ CForLoop
                    { cForVar = loopInfo.astIterator
                    , cForUpperBound = loopInfo.extentParam
                    , cForBody =
                        [ CAssign
                            { cAssignTarget = target
                            , cAssignExpr = rhsExpr
                            }
                        ]
                    }
                ]
            }

lowerAssign :: GenProgram -> Either CAstError (CTensorRef, CExpr, [CTensorName])
lowerAssign genProgram =
    case genProgram.stmtBinding.frontendStmt of
        Reduction{} -> Left ErrCAstReductionNotSupported
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
                        ( CTensorRef
                            { tensorName = outputTensorName
                            , tensorIndices = [singleStoreIndex]
                            }
                        , rhsExpr
                        , tensorArgs
                        )
                _ -> Left $ ErrCAstExpectedSingleStoreIndex (length storeIndices)

lowerExpr :: Map.Map FrontendIterName AstIterVar -> Expr -> Either CAstError CExpr
lowerExpr iterMap expr =
    case expr of
        EConst value -> pure $ CInt value
        ELoad tensor indices -> do
            loweredIndices <- traverse (lowerIxExpr iterMap) indices
            case loweredIndices of
                [_] -> pure $ CLoad (fromString $ tensorNameToString tensor) loweredIndices
                _ -> Left $ ErrCAstExpectedSingleLoadIndex (length loweredIndices)
        EAdd lhs rhs -> CAdd <$> lowerExpr iterMap lhs <*> lowerExpr iterMap rhs
        EMul lhs rhs -> CMul <$> lowerExpr iterMap lhs <*> lowerExpr iterMap rhs

lowerIxExpr :: Map.Map FrontendIterName AstIterVar -> IxExpr -> Either CAstError CExpr
lowerIxExpr iterMap ixExpr =
    case ixExpr of
        IxVar iterName ->
            let iterKey = fromString $ iterNameToString iterName
             in case Map.lookup iterKey iterMap of
                    Just loweredName -> pure $ CVar loweredName
                    Nothing -> Left $ ErrCAstUnknownIterBinding iterKey

collectLoadTensors :: Expr -> [CTensorName]
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
