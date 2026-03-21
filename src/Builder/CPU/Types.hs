module Builder.CPU.Types (
    CompiledSharedObject (..),
) where

import           Runtime.Types (KernelSignature)

data CompiledSharedObject = CompiledSharedObject
    { cSourcePath :: FilePath
    , sharedObjectPath :: FilePath
    , kernelSignature :: KernelSignature
    }
    deriving (Eq, Show)
