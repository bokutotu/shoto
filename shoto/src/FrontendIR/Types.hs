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
import           Data.String        (IsString (fromString))

newtype IterName = IterName String
    deriving (Eq, Ord, Show)

newtype ParamName = ParamName String
    deriving (Eq, Ord, Show)

newtype TensorName = TensorName String
    deriving (Eq, Ord, Show)

instance IsString IterName where
    fromString = IterName

instance IsString ParamName where
    fromString = ParamName

instance IsString TensorName where
    fromString = TensorName

iterNameToString :: IterName -> String
iterNameToString (IterName name) = name

paramNameToString :: ParamName -> String
paramNameToString (ParamName name) = name

tensorNameToString :: TensorName -> String
tensorNameToString (TensorName name) = name

data Axis = Axis
    { iter :: IterName
    , extent :: ParamName
    }
    deriving (Eq, Show)

data TensorDecl = TensorDecl
    { tensor :: TensorName
    , shape :: [ParamName]
    }
    deriving (Eq, Show)

newtype IxExpr = IxVar IterName
    deriving (Eq, Show)

data Expr
    = EConst Int
    | ELoad TensorName [IxExpr]
    | EAdd Expr Expr
    | EMul Expr Expr
    deriving (Eq, Show)

data ReductionOp
    = ReduceAdd
    deriving (Eq, Show)

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
