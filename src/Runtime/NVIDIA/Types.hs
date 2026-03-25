{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.NVIDIA.Types (
    NVIDIA (..),
    LoadedNvidiaKernel (..),
    DeviceBuffer (..),
    runNVIDIA,
) where

import           Runtime.NVIDIA.Internal        (NVIDIA (..), runNVIDIA)
import           Runtime.NVIDIA.Internal.Memory (DevicePtr)
import           Runtime.NVIDIA.Internal.Module (Function, Module)
import           Runtime.Types                  (KernelSignature)

data LoadedNvidiaKernel s = LoadedNvidiaKernel
    { loadedKernelSignature :: KernelSignature
    , loadedModule :: Module s
    , loadedFunction :: Function s
    }

instance Show (LoadedNvidiaKernel s) where
    show loadedNvidiaKernel =
        "LoadedNvidiaKernel { loadedKernelSignature = "
            <> show loadedNvidiaKernel.loadedKernelSignature
            <> " }"

data DeviceBuffer s = DeviceBuffer
    { deviceBufferPtr :: DevicePtr s
    , deviceBufferElements :: Int
    }

instance Show (DeviceBuffer s) where
    show deviceBuffer =
        "DeviceBuffer { deviceBufferElements = " <> show deviceBuffer.deviceBufferElements <> " }"
