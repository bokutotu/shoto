{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE RecordWildCards          #-}
{-# LANGUAGE ScopedTypeVariables      #-}

module Runtime (withKernel, allocGpu, copyToGpu, copyToCpu, GpuPtr (..)) where

import           Control.Monad              (when)
import           Foreign                    (Storable (peek, sizeOf),
                                             allocaArray, malloc, peekArray,
                                             withArray)
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.Ptr
import           System.Posix.DynamicLinker

foreign import ccall "cudaMalloc" cudaMalloc :: Ptr (Ptr ()) -> CSize -> IO CInt

foreign import ccall "cudaMemcpy" cudaMemcpy :: Ptr () -> Ptr () -> CSize -> CInt -> IO CInt

foreign import ccall "&cudaFree" cudaFree :: FinalizerPtr a

withKernel :: FilePath -> String -> (FunPtr a -> IO b) -> IO b
withKernel soPath funcName callback =
    withDL soPath [RTLD_NOW] $ \dl -> do
        funcPtr <- dlsym dl funcName
        callback funcPtr

data GpuPtr a = GpuPtr {ptr :: ForeignPtr a, size :: Int}

allocGpu :: forall a. (Storable a) => Int -> IO (GpuPtr a)
allocGpu size = do
    let bytes = fromIntegral $ size * sizeOf (undefined :: a)
    ptr <- malloc
    result <- cudaMalloc ptr bytes
    when (result /= 0) $ error $ "cudaMalloc Failed: " ++ show result
    devicePtr <- peek ptr
    fptr <- newForeignPtr cudaFree devicePtr
    pure $ GpuPtr (castForeignPtr fptr) size

cudaMemcpyHostToDevice :: CInt
cudaMemcpyHostToDevice = 1

cudaMemcpyDeviceToHost :: CInt
cudaMemcpyDeviceToHost = 2

copyToGpu :: forall a. (Storable a) => [a] -> IO (GpuPtr a)
copyToGpu cpu = do
    let size = length cpu
        bytes = fromIntegral $ size * sizeOf (undefined :: a)
    gpuPtr@GpuPtr{ptr} <- allocGpu size
    withArray cpu $ \cpuPtr -> withForeignPtr ptr $ \devicePtr -> do
        result <- cudaMemcpy (castPtr devicePtr) (castPtr cpuPtr) bytes cudaMemcpyHostToDevice
        when (result /= 0) $ error $ "cudaMemcpy failed: " ++ show result
    pure gpuPtr

copyToCpu :: forall a. (Storable a) => GpuPtr a -> IO [a]
copyToCpu GpuPtr{..} = do
    let bytes = fromIntegral $ size * sizeOf (undefined :: a)
    allocaArray size $ \cpuPtr -> do
        withForeignPtr ptr $ \devicePtr -> do
            result <- cudaMemcpy (castPtr cpuPtr) (castPtr devicePtr) bytes cudaMemcpyDeviceToHost
            when (result /= 0) $ error $ "cudaMemcpy failed: " ++ show result
        peekArray size cpuPtr
