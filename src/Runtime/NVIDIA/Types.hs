{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.NVIDIA.Types (
    NVIDIA (..),
    LoadedNvidiaKernel (..),
    DeviceBuffer (..),
    runNVIDIA,
) where

import           Codegen.CUDA.Ast               (CudaDim)
import           Runtime.NVIDIA.Internal        (NVIDIA (..), runNVIDIA)
import           Runtime.NVIDIA.Internal.Memory (DevicePtr)
import           Runtime.NVIDIA.Internal.Module (Function, Module)
import           Runtime.Types                  (KernelSignature)

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
