{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot   #-}

module Codegen.GenIR (
    GenIRError (..),
    AstIterVar (..),
    ExtentParamName (..),
    GenTensorDecl (..),
    GenTensorRef (..),
    GenExpr (..),
    GenStmt (..),
    GenProgram (..),
    buildGenProgram,
) where

import qualified Data.List.NonEmpty  as NE
import qualified Data.Map.Strict     as Map
import           Data.String         (IsString (fromString))
import           FrontendIR.Types    (Axis (..), Expr (..), IterName,
                                      IxExpr (..), Program (..), ReductionOp,
                                      Stmt (..), TensorDecl (..), TensorName,
                                      paramNameToString)
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

newtype AstIterVar = AstIterVar String deriving (Eq, Ord, Show)

instance IsString AstIterVar where fromString = AstIterVar

newtype ExtentParamName = ExtentParamName String deriving (Eq, Ord, Show)

instance IsString ExtentParamName where fromString = ExtentParamName

data GenTensorDecl = GenTensorDecl
    { genTensor :: TensorName
    , genShape :: [ExtentParamName]
    }
    deriving (Eq, Show)

data GenTensorRef = GenTensorRef
    { genTensor :: TensorName
    , genIndices :: [AstIterVar]
    }
    deriving (Eq, Show)

data GenExpr
    = GenConst Int
    | GenLoad GenTensorRef
    | GenAdd GenExpr GenExpr
    | GenMul GenExpr GenExpr
    deriving (Eq, Show)

data GenStmt
    = GenFor
        { genIter :: AstIterVar
        , genBound :: ExtentParamName
        , genBody :: [GenStmt]
        }
    | GenAssign
        { genTarget :: GenTensorRef
        , genExpr :: GenExpr
        }
    | GenReduction
        { genReductionOp :: ReductionOp
        , genTarget :: GenTensorRef
        , genExpr :: GenExpr
        }
    deriving (Eq, Show)

data GenProgram = GenProgram
    { genExtentParams :: [ExtentParamName]
    , genTensorDecls :: [GenTensorDecl]
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

buildExtentParams :: Program -> [ExtentParamName]
buildExtentParams program =
    fromString . paramNameToString . (.extent) <$> NE.toList program.axes

buildTensorDecls :: Program -> [GenTensorDecl]
buildTensorDecls program =
    [ GenTensorDecl
        { genTensor = tensorDecl.tensor
        , genShape = fromString . paramNameToString <$> tensorDecl.shape
        }
    | tensorDecl <- NE.toList program.tensors
    ]

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
                bound <- parseUpperBound (fromString iterName) cond
                loweredBody <- lowerNestedTree env body
                pure
                    GenFor
                        { genIter = fromString iterName
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

buildIterSubst :: [Axis] -> [AstIterVar] -> Map.Map IterName AstIterVar
buildIterSubst axes stmtArgs =
    Map.fromList $ zip ((.iter) <$> axes) stmtArgs

lowerFrontendStmt ::
    Map.Map IterName AstIterVar ->
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
    Map.Map IterName AstIterVar ->
    TensorName ->
    [IxExpr] ->
    Either GenIRError GenTensorRef
lowerTensorRef iterSubst tensor indices =
    GenTensorRef tensor <$> traverse (lowerIxExpr iterSubst) indices

lowerExpr :: Map.Map IterName AstIterVar -> Expr -> Either GenIRError GenExpr
lowerExpr iterSubst expr =
    case expr of
        EConst value -> pure $ GenConst value
        ELoad tensor indices -> GenLoad <$> lowerTensorRef iterSubst tensor indices
        EAdd lhs rhs -> GenAdd <$> lowerExpr iterSubst lhs <*> lowerExpr iterSubst rhs
        EMul lhs rhs -> GenMul <$> lowerExpr iterSubst lhs <*> lowerExpr iterSubst rhs

lowerIxExpr :: Map.Map IterName AstIterVar -> IxExpr -> Either GenIRError AstIterVar
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

parseUpperBound :: AstIterVar -> AstExpression -> Either GenIRError ExtentParamName
parseUpperBound loopIter cond =
    let AstIterVar loopIterName = loopIter
     in case cond of
            ExprOp (OpLt (ExprId lhs) (ExprId rhs))
                | lhs == loopIterName -> pure $ fromString rhs
                | otherwise -> Left ErrGenExpectedLoopUpperBound
            _ -> Left ErrGenExpectedLoopUpperBound

parseStmtCallExpr :: AstExpression -> Either GenIRError [AstIterVar]
parseStmtCallExpr expr =
    case expr of
        ExprOp (OpCall (ExprId _) args) ->
            traverse parseCallArg args
        _ -> Left ErrGenUnsupportedStmtCall
  where
    parseCallArg arg =
        case arg of
            ExprId indexVar -> Right $ fromString indexVar
            _ -> Left ErrGenExpectedCallArgumentIdentifier
