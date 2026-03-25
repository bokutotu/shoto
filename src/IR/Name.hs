module IR.Name (
    IterName (..),
    ParamName (..),
    TensorName (..),
    KernelName (..),
    StmtName (..),
    iterNameToString,
    paramNameToString,
    tensorNameToString,
    kernelNameToString,
    stmtNameToString,
) where

import           Data.String (IsString (fromString))

newtype IterName = IterName String deriving (Eq, Ord, Show)

newtype ParamName = ParamName String deriving (Eq, Ord, Show)

newtype TensorName = TensorName String deriving (Eq, Ord, Show)

newtype KernelName = KernelName String deriving (Eq, Ord, Show)

newtype StmtName = StmtName String deriving (Eq, Ord, Show)

instance IsString IterName where fromString = IterName

instance IsString ParamName where fromString = ParamName

instance IsString TensorName where fromString = TensorName

instance IsString KernelName where fromString = KernelName

instance IsString StmtName where fromString = StmtName

iterNameToString :: IterName -> String
iterNameToString (IterName name) = name

paramNameToString :: ParamName -> String
paramNameToString (ParamName name) = name

tensorNameToString :: TensorName -> String
tensorNameToString (TensorName name) = name

kernelNameToString :: KernelName -> String
kernelNameToString (KernelName name) = name

stmtNameToString :: StmtName -> String
stmtNameToString (StmtName name) = name
