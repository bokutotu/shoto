{-# LANGUAGE OverloadedRecordDot #-}

module CUDA.Memory (
    DevicePtr (..),
    allocBytes,
    freeDevicePtr,
    copyBytesToDevice,
    copyBytesFromDevice,
    copyBytesToDeviceFromForeignPtr,
    copyBytesFromDeviceToForeignPtr,
) where

import CUDA.Core (
    CUDA,
    CudaError (..),
    expectDriverSuccess,
    throwCUDA,
 )
import CUDA.Internal.Driver.FFI
import Control.Monad.IO.Class (liftIO)
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr)
import Foreign.Storable (peek)

newtype DevicePtr s = DevicePtr
    { rawDevicePtr :: CuDevicePtr
    }
    deriving (Eq, Show)

allocBytes :: Int -> CUDA s (DevicePtr s)
allocBytes bytes
    | bytes < 0 = throwCUDA $ CudaUsageError "allocBytes: negative byte count"
    | otherwise = do
        (result, devicePtr) <-
            liftIO $
                alloca $ \devicePtrPtr -> do
                    result <- c_cuMemAlloc devicePtrPtr (fromIntegral bytes)
                    devicePtr <- peek devicePtrPtr
                    pure (result, devicePtr)
        expectDriverSuccess "cuMemAlloc_v2" result
        pure DevicePtr{rawDevicePtr = devicePtr}

freeDevicePtr :: DevicePtr s -> CUDA s ()
freeDevicePtr devicePtr =
    expectDriverSuccess "cuMemFree_v2" =<< liftIO (c_cuMemFree devicePtr.rawDevicePtr)

copyBytesToDevice :: Ptr a -> Int -> DevicePtr s -> CUDA s ()
copyBytesToDevice hostPtr byteCount devicePtr
    | byteCount < 0 = throwCUDA $ CudaUsageError "copyBytesToDevice: negative byte count"
    | otherwise =
        expectDriverSuccess "cuMemcpyHtoD_v2"
            =<< liftIO
                (c_cuMemcpyHtoD devicePtr.rawDevicePtr hostPtr (fromIntegral byteCount))

copyBytesFromDevice :: DevicePtr s -> Int -> Ptr a -> CUDA s ()
copyBytesFromDevice devicePtr byteCount hostPtr
    | byteCount < 0 = throwCUDA $ CudaUsageError "copyBytesFromDevice: negative byte count"
    | otherwise =
        expectDriverSuccess "cuMemcpyDtoH_v2"
            =<< liftIO
                (c_cuMemcpyDtoH hostPtr devicePtr.rawDevicePtr (fromIntegral byteCount))

copyBytesToDeviceFromForeignPtr :: ForeignPtr a -> Int -> DevicePtr s -> CUDA s ()
copyBytesToDeviceFromForeignPtr hostBuffer byteCount devicePtr
    | byteCount < 0 =
        throwCUDA $ CudaUsageError "copyBytesToDeviceFromForeignPtr: negative byte count"
    | otherwise =
        expectDriverSuccess "cuMemcpyHtoD_v2"
            =<< liftIO
                ( withForeignPtr hostBuffer $ \hostPtr ->
                    c_cuMemcpyHtoD devicePtr.rawDevicePtr hostPtr (fromIntegral byteCount)
                )

copyBytesFromDeviceToForeignPtr :: DevicePtr s -> Int -> ForeignPtr a -> CUDA s ()
copyBytesFromDeviceToForeignPtr devicePtr byteCount hostBuffer
    | byteCount < 0 =
        throwCUDA $ CudaUsageError "copyBytesFromDeviceToForeignPtr: negative byte count"
    | otherwise =
        expectDriverSuccess "cuMemcpyDtoH_v2"
            =<< liftIO
                ( withForeignPtr hostBuffer $ \hostPtr ->
                    c_cuMemcpyDtoH hostPtr devicePtr.rawDevicePtr (fromIntegral byteCount)
                )
