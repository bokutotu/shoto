module IR.Tensor (
    TensorDecl (..),
    TensorRef (..),
) where

import           IR.Name (ParamName, TensorName)

data TensorDecl = TensorDecl
    { tensor :: TensorName
    , shape :: [ParamName]
    }
    deriving (Eq, Show)

data TensorRef ix = TensorRef
    { tensorName :: TensorName
    , tensorIndices :: [ix]
    }
    deriving (Eq, Show)
