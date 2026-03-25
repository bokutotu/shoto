module FrontendIR.Types (
    IterName,
    ParamName,
    TensorName,
    iterNameToString,
    paramNameToString,
    tensorNameToString,
    Axis (..),
    TensorDecl (..),
    IxExpr (..),
    Expr (..),
    ReductionOp (..),
    Stmt (..),
    Program (..),
    FrontendError (..),
) where

import           Data.List.NonEmpty (NonEmpty)
import           IR.Name            (IterName, ParamName, TensorName,
                                     iterNameToString, paramNameToString,
                                     tensorNameToString)
import           IR.Tensor          (TensorDecl (..))

data Axis = Axis
    { iter :: IterName
    , extent :: ParamName
    }
    deriving (Eq, Show)

newtype IxExpr = IxVar IterName deriving (Eq, Show)

data Expr
    = EConst Int
    | ELoad TensorName [IxExpr]
    | EAdd Expr Expr
    | EMul Expr Expr
    deriving (Eq, Show)

data ReductionOp = ReduceAdd deriving (Eq, Show)

data Stmt
    = Assign
        { outputTensor :: TensorName
        , outputIndex :: [IxExpr]
        , rhs :: Expr
        }
    | Reduction
        { reductionOp :: ReductionOp
        , outputTensor :: TensorName
        , outputIndex :: [IxExpr]
        , rhs :: Expr
        }
    deriving (Eq, Show)

data Program = Program
    { axes :: NonEmpty Axis
    , tensors :: NonEmpty TensorDecl
    , stmts :: NonEmpty Stmt
    }
    deriving (Eq, Show)

data FrontendError
    = ErrDuplicateIter IterName
    | ErrDuplicateParam ParamName
    | ErrDuplicateTensor TensorName
    | ErrUndeclaredTensor TensorName
    | ErrTensorRankMismatch TensorName Int Int
    | ErrUnknownTensorShapeParam TensorName ParamName
    | ErrUnknownIndexIter IterName
    | ErrStoreIndexMismatch [IterName] [IterName]
    | ErrReductionOutputNotSubsequence [IterName] [IterName]
    | ErrReductionRequiresReducedAxis
    | ErrMultiStmtRequiresSingleReductionAxis [IterName]
    | ErrLoadIndexMismatch TensorName [IterName] [IterName]
    deriving (Eq, Show)
