{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.NVIDIA.Types (
    NVIDIA (..),
    CompiledCudaProgram (..),
    LoadedNvidiaKernel (..),
    DeviceBuffer (..),
    runNVIDIA,
) where

import           Codegen.CUDA.Ast               (CudaDim)
import qualified Data.ByteString                as BS
import           Runtime.NVIDIA.Internal        (NVIDIA (..), runNVIDIA)
import           Runtime.NVIDIA.Internal.Memory (DevicePtr)
import           Runtime.NVIDIA.Internal.Module (Function, Module)
import           Runtime.Types                  (KernelSignature)

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

data LoadedNvidiaKernel s = LoadedNvidiaKernel
    { loadedKernelSignature :: KernelSignature
    , loadedCudaDim :: CudaDim
    , loadedModule :: Module s
    , loadedFunction :: Function s
    }

instance Show (LoadedNvidiaKernel s) where
    show loadedNvidiaKernel =
        "LoadedNvidiaKernel { loadedKernelSignature = "
            <> show loadedNvidiaKernel.loadedKernelSignature
            <> ", loadedCudaDim = "
            <> show loadedNvidiaKernel.loadedCudaDim
            <> " }"

data DeviceBuffer s = DeviceBuffer
    { deviceBufferPtr :: DevicePtr s
    , deviceBufferElements :: Int
    }

instance Show (DeviceBuffer s) where
    show deviceBuffer =
        "DeviceBuffer { deviceBufferElements = " <> show deviceBuffer.deviceBufferElements <> " }"
