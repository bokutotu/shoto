{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot   #-}

module Codegen.GenIR (
    GenIRError (..),
    GenExpr (..),
    GenStmt (..),
    GenProgram (..),
    buildGenProgram,
) where

import qualified Data.List.NonEmpty  as NE
import qualified Data.Map.Strict     as Map
import           FrontendIR.Types    (Axis (..), Expr (..), IxExpr (..),
                                      Program (..), ReductionOp, Stmt (..))
import           IR.Name             (IterName (..), ParamName (..), TensorName,
                                      iterNameToString)
import           IR.Tensor           (TensorDecl (..), TensorRef (..))
import           Polyhedral.Internal (AstExpression (..), AstOp (..),
                                      AstTree (..))

data GenIRError
    = ErrGenExpectedTopLevelFor
    | ErrGenExpectedCanonicalFor
    | ErrGenMalformedLoopNest
    | ErrGenExpectedLoopUpperBound
    | ErrGenUnsupportedStmtCall
    | ErrGenExpectedCallArgumentIdentifier
    | ErrGenMissingStmtArgument IterName
    deriving (Eq, Show)

data GenExpr
    = GenConst Int
    | GenLoad (TensorRef IterName)
    | GenAdd GenExpr GenExpr
    | GenMul GenExpr GenExpr
    deriving (Eq, Show)

data GenStmt
    = GenFor
        { genIter :: IterName
        , genBound :: ParamName
        , genBody :: [GenStmt]
        }
    | GenAssign
        { genTarget :: TensorRef IterName
        , genExpr :: GenExpr
        }
    | GenReduction
        { genReductionOp :: ReductionOp
        , genTarget :: TensorRef IterName
        , genExpr :: GenExpr
        }
    deriving (Eq, Show)

data GenProgram = GenProgram
    { genExtentParams :: [ParamName]
    , genTensorDecls :: [TensorDecl]
    , genBody :: [GenStmt]
    }
    deriving (Eq, Show)

data BuildEnv = BuildEnv
    { frontendAxes :: [Axis]
    , frontendStmt :: Stmt
    }

buildGenProgram :: AstTree -> Program -> Either GenIRError GenProgram
buildGenProgram ast program =
    case unwrapTree ast of
        AstFor{} -> do
            topLevelStmt <- lowerTree buildEnv ast
            pure
                GenProgram
                    { genExtentParams = buildExtentParams program
                    , genTensorDecls = buildTensorDecls program
                    , genBody = [topLevelStmt]
                    }
        _ -> Left ErrGenExpectedTopLevelFor
  where
    buildEnv =
        BuildEnv
            { frontendAxes = NE.toList program.axes
            , frontendStmt = NE.head program.stmts
            }

buildExtentParams :: Program -> [ParamName]
buildExtentParams program =
    (.extent) <$> NE.toList program.axes

buildTensorDecls :: Program -> [TensorDecl]
buildTensorDecls program =
    NE.toList program.tensors

lowerTree :: BuildEnv -> AstTree -> Either GenIRError GenStmt
lowerTree env tree =
    case unwrapTree tree of
        AstFor
            { forIterator = iterName
            , forInit = ExprInt 0
            , forInc = ExprInt 1
            , forCond = cond
            , forBody = body
            } -> do
                bound <- parseUpperBound (IterName iterName) cond
                loweredBody <- lowerNestedTree env body
                pure
                    GenFor
                        { genIter = IterName iterName
                        , genBound = bound
                        , genBody = [loweredBody]
                        }
        AstFor{} -> Left ErrGenExpectedCanonicalFor
        AstUser expr -> do
            stmtArgs <- parseStmtCallExpr expr
            lowerFrontendStmt
                (buildIterSubst env.frontendAxes stmtArgs)
                env.frontendStmt
        _ -> Left ErrGenMalformedLoopNest

lowerNestedTree :: BuildEnv -> AstTree -> Either GenIRError GenStmt
lowerNestedTree env tree =
    case unwrapTree tree of
        AstFor{} -> lowerTree env tree
        AstUser{} -> lowerTree env tree
        _ -> Left ErrGenMalformedLoopNest

buildIterSubst :: [Axis] -> [IterName] -> Map.Map IterName IterName
buildIterSubst axes stmtArgs =
    Map.fromList $ zip ((.iter) <$> axes) stmtArgs

lowerFrontendStmt ::
    Map.Map IterName IterName ->
    Stmt ->
    Either GenIRError GenStmt
lowerFrontendStmt iterSubst stmt =
    case stmt of
        Assign{} ->
            GenAssign
                <$> lowerTensorRef iterSubst stmt.outputTensor stmt.outputIndex
                <*> lowerExpr iterSubst stmt.rhs
        Reduction{} ->
            GenReduction
                <$> pure stmt.reductionOp
                <*> lowerTensorRef iterSubst stmt.outputTensor stmt.outputIndex
                <*> lowerExpr iterSubst stmt.rhs

lowerTensorRef ::
    Map.Map IterName IterName ->
    TensorName ->
    [IxExpr] ->
    Either GenIRError (TensorRef IterName)
lowerTensorRef iterSubst tensor indices =
    TensorRef tensor <$> traverse (lowerIxExpr iterSubst) indices

lowerExpr :: Map.Map IterName IterName -> Expr -> Either GenIRError GenExpr
lowerExpr iterSubst expr =
    case expr of
        EConst value -> pure $ GenConst value
        ELoad tensor indices -> GenLoad <$> lowerTensorRef iterSubst tensor indices
        EAdd lhs rhs -> GenAdd <$> lowerExpr iterSubst lhs <*> lowerExpr iterSubst rhs
        EMul lhs rhs -> GenMul <$> lowerExpr iterSubst lhs <*> lowerExpr iterSubst rhs

lowerIxExpr :: Map.Map IterName IterName -> IxExpr -> Either GenIRError IterName
lowerIxExpr iterSubst ixExpr =
    case ixExpr of
        IxVar iterName ->
            case Map.lookup iterName iterSubst of
                Just astIter -> pure astIter
                Nothing -> Left $ ErrGenMissingStmtArgument iterName

unwrapTree :: AstTree -> AstTree
unwrapTree tree =
    case tree of
        AstMark _ inner -> unwrapTree inner
        AstBlock [single] -> unwrapTree single
        _ -> tree

parseUpperBound :: IterName -> AstExpression -> Either GenIRError ParamName
parseUpperBound loopIter cond =
    let loopIterName = iterNameToString loopIter
     in case cond of
            ExprOp (OpLt (ExprId lhs) (ExprId rhs))
                | lhs == loopIterName -> pure $ ParamName rhs
                | otherwise -> Left ErrGenExpectedLoopUpperBound
            _ -> Left ErrGenExpectedLoopUpperBound

parseStmtCallExpr :: AstExpression -> Either GenIRError [IterName]
parseStmtCallExpr expr =
    case expr of
        ExprOp (OpCall (ExprId _) args) ->
            traverse parseCallArg args
        _ -> Left ErrGenUnsupportedStmtCall
  where
    parseCallArg arg =
        case arg of
            ExprId indexVar -> Right $ IterName indexVar
            _ -> Left ErrGenExpectedCallArgumentIdentifier
