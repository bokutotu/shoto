{-# LANGUAGE OverloadedRecordDot #-}

module Builder.NVIDIA.Types (
    CompiledCudaProgram (..),
) where

import           Codegen.CUDA.Ast (CudaDim)
import qualified Data.ByteString  as BS
import           Runtime.Types    (KernelSignature)

data CompiledCudaProgram = CompiledCudaProgram
    { compiledPtx :: BS.ByteString
    , compiledKernelSignature :: KernelSignature
    , compiledCudaDim :: CudaDim
    }

instance Show CompiledCudaProgram where
    show compiledCudaProgram =
        "CompiledCudaProgram { compiledKernelSignature = "
            <> show compiledCudaProgram.compiledKernelSignature
            <> ", compiledCudaDim = "
            <> show compiledCudaProgram.compiledCudaDim
            <> " }"
