{-# LANGUAGE OverloadedRecordDot #-}

module Builder.NVIDIA.Types (
    CompiledCudaProgram (..),
) where

import qualified Data.ByteString as BS
import           Runtime.Types   (KernelSignature)

data CompiledCudaProgram = CompiledCudaProgram
    { compiledPtx :: BS.ByteString
    , compiledKernelSignature :: KernelSignature
    }

instance Show CompiledCudaProgram where
    show compiledCudaProgram =
        "CompiledCudaProgram { compiledKernelSignature = "
            <> show compiledCudaProgram.compiledKernelSignature
            <> " }"
