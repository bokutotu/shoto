{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.NVIDIA.Internal.Module (
    Module (..),
    Function (..),
    Dim3 (..),
    KernelArg (..),
    loadModuleData,
    unloadModule,
    getFunction,
    launchKernel,
    synchronize,
) where

import           Control.Monad.IO.Class             (liftIO)
import qualified Data.ByteString                    as BS
import qualified Data.ByteString.Unsafe             as BSU
import           Data.Word                          (Word32)
import           Foreign.C.String                   (withCString)
import           Foreign.C.Types                    (CInt (..), CUInt (..))
import           Foreign.Marshal.Alloc              (alloca)
import           Foreign.Marshal.Array              (withArray)
import           Foreign.Marshal.Utils              (with)
import           Foreign.Ptr                        (Ptr, castPtr, nullPtr)
import           Foreign.Storable                   (peek)
import           Runtime.NVIDIA.Internal.Core       (CUDA, expectDriverSuccess)
import           Runtime.NVIDIA.Internal.Driver.FFI
import           Runtime.NVIDIA.Internal.Memory     (DevicePtr (..))

newtype Module s = Module
    { rawModule :: RawModule
    }
    deriving (Eq, Show)

data Function s = Function
    { rawFunction :: RawFunction
    , ownerModule :: Module s
    }

instance Show (Function s) where
    show _ = "Function"

data Dim3 = Dim3
    { dimX :: Word32
    , dimY :: Word32
    , dimZ :: Word32
    }
    deriving (Eq, Show)

data KernelArg s
    = KernelArgInt Int
    | KernelArgDevicePtr (DevicePtr s)

loadModuleData :: BS.ByteString -> CUDA s (Module s)
loadModuleData imageBytes = do
    (result, rawModule) <-
        liftIO $
            BSU.unsafeUseAsCString imageBytes $ \imagePtr ->
                alloca $ \modulePtr -> do
                    result <- c_cuModuleLoadData modulePtr (castPtr imagePtr)
                    rawModule <- peek modulePtr
                    pure (result, rawModule)
    expectDriverSuccess "cuModuleLoadData" result
    pure Module{rawModule}

unloadModule :: Module s -> CUDA s ()
unloadModule loadedModule =
    expectDriverSuccess "cuModuleUnload"
        =<< liftIO (c_cuModuleUnload loadedModule.rawModule)

getFunction :: Module s -> String -> CUDA s (Function s)
getFunction loadedModule symbolName = do
    (result, rawFunction) <-
        liftIO $
            withCString symbolName $ \symbolPtr ->
                alloca $ \functionPtr -> do
                    result <- c_cuModuleGetFunction functionPtr loadedModule.rawModule symbolPtr
                    rawFunction <- peek functionPtr
                    pure (result, rawFunction)
    expectDriverSuccess "cuModuleGetFunction" result
    pure Function{rawFunction, ownerModule = loadedModule}

launchKernel :: Function s -> Dim3 -> Dim3 -> [KernelArg s] -> CUDA s ()
launchKernel function gridDim blockDim kernelArgs = do
    result <-
        liftIO $
            withKernelArgPointers kernelArgs $ \kernelArgPointers ->
                withArray kernelArgPointers $ \kernelArgVector ->
                    c_cuLaunchKernel
                        function.rawFunction
                        (toCUInt gridDim.dimX)
                        (toCUInt gridDim.dimY)
                        (toCUInt gridDim.dimZ)
                        (toCUInt blockDim.dimX)
                        (toCUInt blockDim.dimY)
                        (toCUInt blockDim.dimZ)
                        0
                        nullPtr
                        kernelArgVector
                        nullPtr
    expectDriverSuccess "cuLaunchKernel" result

synchronize :: CUDA s ()
synchronize =
    expectDriverSuccess "cuCtxSynchronize"
        =<< liftIO c_cuCtxSynchronize

withKernelArgPointers :: [KernelArg s] -> ([Ptr ()] -> IO a) -> IO a
withKernelArgPointers kernelArgs continue =
    go kernelArgs []
  where
    go [] reversedPointers =
        continue $ reverse reversedPointers
    go (kernelArg : remainingArgs) reversedPointers =
        case kernelArg of
            KernelArgInt value ->
                with (fromIntegral value :: CInt) $ \valuePtr ->
                    go remainingArgs (castPtr valuePtr : reversedPointers)
            KernelArgDevicePtr devicePtr ->
                with devicePtr.rawDevicePtr $ \devicePtrPtr ->
                    go remainingArgs (castPtr devicePtrPtr : reversedPointers)

toCUInt :: Word32 -> CUInt
toCUInt = fromIntegral
