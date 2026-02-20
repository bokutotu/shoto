module FrontendIR.Types (
    IterName,
    ParamName,
    TensorName,
    iterNameToString,
    paramNameToString,
    tensorNameToString,
    Axis (..),
    IxExpr (..),
    Expr (..),
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

newtype IxExpr = IxVar IterName
    deriving (Eq, Show)

data Expr
    = EConst Int
    | ELoad TensorName [IxExpr]
    | EAdd Expr Expr
    | EMul Expr Expr
    deriving (Eq, Show)

data Stmt = Stmt
    { outputTensor :: TensorName
    , outputIndex :: [IxExpr]
    , rhs :: Expr
    }
    deriving (Eq, Show)

data Program = Program
    { axes :: NonEmpty Axis
    , stmt :: Stmt
    }
    deriving (Eq, Show)

data FrontendError
    = ErrDuplicateIter IterName
    | ErrDuplicateParam ParamName
    | ErrStoreIndexMismatch [IterName] [IterName]
    | ErrLoadIndexMismatch TensorName [IterName] [IterName]
    deriving (Eq, Show)
