{-# LANGUAGE CPP                        #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedRecordDot        #-}
{-# LANGUAGE RankNTypes                 #-}

module Runtime.NVIDIA.Types (
    NVIDIA (..),
    CompiledCudaProgram (..),
    LoadedNvidiaKernel (..),
    DeviceBuffer (..),
    runNVIDIA,
    liftCuda,
) where

import           Codegen.CUDA.Ast               (CudaDim)
import           Control.Monad.Except           (ExceptT, MonadError,
                                                 runExceptT)
#ifdef CUDA_RUNTIME
import           Control.Monad.IO.Class         (MonadIO)
import           Control.Monad.Trans            (lift)
import qualified Data.ByteString                as BS
import           Runtime.NVIDIA.Internal        (CUDA, CudaError (..), runCUDA)
import           Runtime.NVIDIA.Internal.Memory (DevicePtr)
import           Runtime.NVIDIA.Internal.Module (Function, Module)
#else
import           Control.Monad.IO.Class         (MonadIO, liftIO)
#endif
import           Runtime.Types                  (KernelSignature,
                                                 RuntimeError (..))

#ifdef CUDA_RUNTIME
newtype NVIDIA s a = NVIDIA {unNVIDIA :: ExceptT RuntimeError (CUDA s) a}
    deriving (Functor, Applicative, Monad, MonadIO, MonadError RuntimeError)

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

runNVIDIA :: (forall s. NVIDIA s a) -> IO (Either RuntimeError a)
runNVIDIA action = do
    cudaResult <- runCUDA (runExceptT (action.unNVIDIA))
    pure $
        case cudaResult of
            Left cudaError -> Left $ mapCudaError cudaError
            Right runtimeResult -> runtimeResult

liftCuda :: CUDA s a -> NVIDIA s a
liftCuda = NVIDIA . lift

mapCudaError :: CudaError -> RuntimeError
mapCudaError cudaError =
    case cudaError of
        CudaDriverError{cudaFunction, cudaCode, cudaName, cudaMessage} ->
            ErrRuntimeCudaDriverError cudaFunction cudaCode cudaName cudaMessage
        CudaNvrtcError{cudaFunction, cudaCode, cudaMessage, cudaLog} ->
            ErrRuntimeCudaNvrtcError cudaFunction cudaCode cudaMessage cudaLog
        CudaUsageError message ->
            ErrRuntimeCudaNvrtcError "cuda-usage" (-1) (Just message) Nothing
#else
newtype NVIDIA s a = NVIDIA {unNVIDIA :: ExceptT RuntimeError IO a}
    deriving (Functor, Applicative, Monad, MonadIO, MonadError RuntimeError)

data CompiledCudaProgram = CompiledCudaProgram
    { compiledKernelSignature :: KernelSignature
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
    }

instance Show (LoadedNvidiaKernel s) where
    show loadedNvidiaKernel =
        "LoadedNvidiaKernel { loadedKernelSignature = "
            <> show loadedNvidiaKernel.loadedKernelSignature
            <> ", loadedCudaDim = "
            <> show loadedNvidiaKernel.loadedCudaDim
            <> " }"

data DeviceBuffer s = DeviceBuffer
    { deviceBufferElements :: Int
    }

instance Show (DeviceBuffer s) where
    show deviceBuffer =
        "DeviceBuffer { deviceBufferElements = " <> show deviceBuffer.deviceBufferElements <> " }"

runNVIDIA :: (forall s. NVIDIA s a) -> IO (Either RuntimeError a)
runNVIDIA action = runExceptT action.unNVIDIA

liftCuda :: IO a -> NVIDIA s a
liftCuda = liftIO
#endif
