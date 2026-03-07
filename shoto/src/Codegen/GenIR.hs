{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.GenIR (
    GenIRError (..),
    AstIterVar (..),
    FrontendIterName (..),
    ExtentParamName (..),
    AstStmtName (..),
    GenLoop (..),
    GenStmtBinding (..),
    GenProgram (..),
    buildGenProgram,
) where

import qualified Data.List.NonEmpty as NE
import           Data.String        (IsString (fromString))
import           FrontendIR.Types   (Axis (..), IxExpr (..), Program (..),
                                     Stmt (..), iterNameToString,
                                     paramNameToString)
import           ISL                (AstExpression (..), AstOp (..),
                                     AstTree (..))

data GenIRError
    = ErrGenExpectedSingleStatement Int
    | ErrGenExpectedSingleAxis Int
    | ErrGenOnlyAssignSupported
    | ErrGenExpectedSingleOutputIndex Int
    | ErrGenExpectedTopLevelFor
    | ErrGenExpectedCanonicalFor
    | ErrGenExpectedLoopUpperBound
    | ErrGenExpectedSingleStmtCall
    | ErrGenExpectedStmtCallExpression
    | ErrGenExpectedSingleCallArgument Int
    | ErrGenExpectedCallArgumentIdentifier
    | ErrGenLoopAndCallIndexMismatch AstIterVar AstIterVar
    | ErrGenAxisAndStmtIterMismatch FrontendIterName FrontendIterName
    | ErrGenLoopAndAxisBoundMismatch ExtentParamName ExtentParamName
    | ErrGenStmtNameMismatch AstStmtName AstStmtName
    deriving (Eq, Show)

newtype AstIterVar = AstIterVar String deriving (Eq, Ord, Show)

instance IsString AstIterVar where fromString = AstIterVar

newtype FrontendIterName = FrontendIterName String deriving (Eq, Ord, Show)

instance IsString FrontendIterName where fromString = FrontendIterName

newtype ExtentParamName = ExtentParamName String deriving (Eq, Ord, Show)

instance IsString ExtentParamName where fromString = ExtentParamName

newtype AstStmtName = AstStmtName String deriving (Eq, Ord, Show)

instance IsString AstStmtName where fromString = AstStmtName

data GenLoop = GenLoop
    { astIterator :: AstIterVar
    , frontendIter :: FrontendIterName
    , extentParam :: ExtentParamName
    }
    deriving (Eq, Show)

data GenStmtBinding = GenStmtBinding
    { astStmtName :: AstStmtName
    , astIndexVar :: AstIterVar
    , frontendStmt :: Stmt
    }
    deriving (Eq, Show)

data GenProgram = GenProgram
    { loop :: GenLoop
    , stmtBinding :: GenStmtBinding
    }
    deriving (Eq, Show)

buildGenProgram :: AstTree -> Program -> Either GenIRError GenProgram
buildGenProgram ast program = do
    axis <- expectSingleAxis program
    stmt <- expectSingleStmt program
    frontendIterName <- expectAssignStmtIter stmt
    (loopIter, loopBound, body) <- parseTopLevelFor ast
    (stmtName, stmtIndexVar) <- parseStmtCall body
    let expectedStmtName = fromString "S0"
    if stmtName == expectedStmtName
        then pure ()
        else Left $ ErrGenStmtNameMismatch expectedStmtName stmtName
    if stmtIndexVar == loopIter
        then pure ()
        else Left $ ErrGenLoopAndCallIndexMismatch loopIter stmtIndexVar
    let axisIterName = fromString $ iterNameToString axis.iter
    if frontendIterName == axisIterName
        then pure ()
        else Left $ ErrGenAxisAndStmtIterMismatch axisIterName frontendIterName
    let axisExtent = fromString $ paramNameToString axis.extent
    if loopBound == axisExtent
        then pure ()
        else Left $ ErrGenLoopAndAxisBoundMismatch axisExtent loopBound
    pure
        GenProgram
            { loop =
                GenLoop
                    { astIterator = loopIter
                    , frontendIter = frontendIterName
                    , extentParam = axisExtent
                    }
            , stmtBinding =
                GenStmtBinding
                    { astStmtName = stmtName
                    , astIndexVar = stmtIndexVar
                    , frontendStmt = stmt
                    }
            }

expectSingleStmt :: Program -> Either GenIRError Stmt
expectSingleStmt program =
    case NE.toList program.stmts of
        [stmt] -> pure stmt
        stmts -> Left $ ErrGenExpectedSingleStatement (length stmts)

expectSingleAxis :: Program -> Either GenIRError Axis
expectSingleAxis program =
    case NE.toList program.axes of
        [axis] -> pure axis
        axes -> Left $ ErrGenExpectedSingleAxis (length axes)

expectAssignStmtIter :: Stmt -> Either GenIRError FrontendIterName
expectAssignStmtIter stmt =
    case stmt of
        Reduction{} -> Left ErrGenOnlyAssignSupported
        Assign{} ->
            case stmt.outputIndex of
                [IxVar iterName] -> pure $ fromString $ iterNameToString iterName
                indices -> Left $ ErrGenExpectedSingleOutputIndex (length indices)

parseTopLevelFor :: AstTree -> Either GenIRError (AstIterVar, ExtentParamName, AstTree)
parseTopLevelFor ast =
    case ast of
        AstMark _ inner -> parseTopLevelFor inner
        AstBlock [single] -> parseTopLevelFor single
        AstFor
            { forIterator = iterName
            , forInit = ExprInt 0
            , forInc = ExprInt 1
            , forCond = cond
            , forBody = body
            } -> do
                let loopIter = fromString iterName
                bound <- parseUpperBound loopIter cond
                pure (loopIter, bound, body)
        AstFor{} -> Left ErrGenExpectedCanonicalFor
        _ -> Left ErrGenExpectedTopLevelFor

parseUpperBound :: AstIterVar -> AstExpression -> Either GenIRError ExtentParamName
parseUpperBound loopIter cond =
    let AstIterVar loopIterName = loopIter
     in case cond of
            ExprOp (OpLt (ExprId lhs) (ExprId rhs))
                | lhs == loopIterName -> pure $ fromString rhs
                | otherwise -> Left ErrGenExpectedLoopUpperBound
            _ -> Left ErrGenExpectedLoopUpperBound

parseStmtCall :: AstTree -> Either GenIRError (AstStmtName, AstIterVar)
parseStmtCall tree =
    case tree of
        AstMark _ inner -> parseStmtCall inner
        AstBlock [single] -> parseStmtCall single
        AstUser expr -> parseStmtCallExpr expr
        _ -> Left ErrGenExpectedSingleStmtCall

parseStmtCallExpr :: AstExpression -> Either GenIRError (AstStmtName, AstIterVar)
parseStmtCallExpr expr =
    case expr of
        ExprOp (OpCall (ExprId stmtName) args) ->
            case args of
                [ExprId indexVar] ->
                    pure
                        ( fromString stmtName
                        , fromString indexVar
                        )
                [ExprInt _] -> Left ErrGenExpectedCallArgumentIdentifier
                [_] -> Left ErrGenExpectedCallArgumentIdentifier
                _ -> Left $ ErrGenExpectedSingleCallArgument (length args)
        _ -> Left ErrGenExpectedStmtCallExpression
