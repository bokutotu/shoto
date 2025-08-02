{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings        #-}
{-# LANGUAGE ScopedTypeVariables      #-}

module Nvrtc where

import           Control.Monad          (when)
import qualified Data.ByteString.Char8  as BS
import           Data.ByteString.Unsafe (unsafeUseAsCString)
import           Foreign
import           Foreign.C.String
import           Foreign.C.Types
import           System.Exit            (exitFailure)
import           Data.IORef
import           System.IO.Unsafe       (unsafePerformIO)

-- NVRTC Types
type NvrtcProgram = Ptr ()

type NvrtcResult = CInt

-- CUDA Driver API Types
type CUdevice = CInt

type CUcontext = Ptr ()

type CUmodule = Ptr ()

type CUfunction = Ptr ()

type CUdeviceptr = CULLong

type CUresult = CInt

type CUstream = Ptr ()

-- Result codes
cudaSuccess, nvrtcSuccess :: CInt
cudaSuccess = 0
nvrtcSuccess = 0

-- NVRTC FFI declarations
foreign import ccall "nvrtcGetErrorString" nvrtcGetErrorString :: NvrtcResult -> IO CString

foreign import ccall "nvrtcCreateProgram"
    nvrtcCreateProgram ::
        Ptr NvrtcProgram -> CString -> CString -> CInt -> Ptr CString -> Ptr CString -> IO NvrtcResult

foreign import ccall "nvrtcCompileProgram"
    nvrtcCompileProgram ::
        NvrtcProgram -> CInt -> Ptr CString -> IO NvrtcResult

foreign import ccall "nvrtcGetProgramLogSize"
    nvrtcGetProgramLogSize ::
        NvrtcProgram -> Ptr CSize -> IO NvrtcResult

foreign import ccall "nvrtcGetProgramLog"
    nvrtcGetProgramLog ::
        NvrtcProgram -> CString -> IO NvrtcResult

foreign import ccall "nvrtcGetPTXSize"
    nvrtcGetPTXSize ::
        NvrtcProgram -> Ptr CSize -> IO NvrtcResult

foreign import ccall "nvrtcGetPTX"
    nvrtcGetPTX ::
        NvrtcProgram -> CString -> IO NvrtcResult

foreign import ccall "nvrtcDestroyProgram"
    nvrtcDestroyProgram ::
        Ptr NvrtcProgram -> IO NvrtcResult

-- CUDA Driver API FFI declarations
foreign import ccall "cuInit" cuInit :: CUInt -> IO CUresult

foreign import ccall "cuDeviceGet" cuDeviceGet :: Ptr CUdevice -> CInt -> IO CUresult

foreign import ccall "cuCtxCreate" cuCtxCreate :: Ptr CUcontext -> CUInt -> CUdevice -> IO CUresult

foreign import ccall "cuCtxDestroy" cuCtxDestroy :: CUcontext -> IO CUresult

foreign import ccall "cuModuleLoadData" cuModuleLoadData :: Ptr CUmodule -> Ptr () -> IO CUresult

foreign import ccall "cuModuleGetFunction"
    cuModuleGetFunction :: Ptr CUfunction -> CUmodule -> CString -> IO CUresult

foreign import ccall "cuMemAlloc" cuMemAlloc :: Ptr CUdeviceptr -> CSize -> IO CUresult

foreign import ccall "cuMemFree" cuMemFree :: CUdeviceptr -> IO CUresult

foreign import ccall "cuMemcpyHtoD" cuMemcpyHtoD :: CUdeviceptr -> Ptr () -> CSize -> IO CUresult

foreign import ccall "cuMemcpyDtoH" cuMemcpyDtoH :: Ptr () -> CUdeviceptr -> CSize -> IO CUresult

foreign import ccall "cuLaunchKernel"
    cuLaunchKernel ::
        CUfunction ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUstream ->
        Ptr (Ptr ()) ->
        Ptr (Ptr ()) ->
        IO CUresult

foreign import ccall "cuCtxSynchronize" cuCtxSynchronize :: IO CUresult

foreign import ccall "cuGetErrorString" cuGetErrorString :: CUresult -> Ptr CString -> IO CUresult

foreign import ccall "cuModuleUnload" cuModuleUnload :: CUmodule -> IO CUresult

-- Helper functions
checkCuda :: String -> CUresult -> IO ()
checkCuda msg result = when (result /= cudaSuccess) $ do
    alloca $ \errStrPtr -> do
        _ <- cuGetErrorString result errStrPtr
        errStr <- peek errStrPtr >>= peekCString
        error $ msg ++ ": " ++ errStr

checkNvrtc :: String -> NvrtcResult -> IO ()
checkNvrtc msg result = when (result /= nvrtcSuccess) $ do
    errStr <- nvrtcGetErrorString result >>= peekCString
    error $ msg ++ ": " ++ errStr

-- Global initialization flag
{-# NOINLINE cudaInitialized #-}
cudaInitialized :: IORef Bool
cudaInitialized = unsafePerformIO $ newIORef False

-- Safe initialization function
ensureCudaInitialized :: IO ()
ensureCudaInitialized = do
    initialized <- readIORef cudaInitialized
    unless initialized $ do
        result <- cuInit 0
        if result == cudaSuccess
            then writeIORef cudaInitialized True
            else checkCuda "cuInit" result
  where
    unless = flip when . not