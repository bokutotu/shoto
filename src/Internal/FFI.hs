{-# LANGUAGE ForeignFunctionInterface #-}

module Internal.FFI (
    -- CUDA Runtime API
    cudaMalloc,
    cudaMemcpy,
    cudaFree,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    
    -- NVRTC Types
    NvrtcProgram,
    NvrtcResult,
    
    -- CUDA Driver API Types
    CUdevice,
    CUcontext,
    CUmodule,
    CUfunction,
    CUdeviceptr,
    CUresult,
    CUstream,
    
    -- Result codes
    cudaSuccess,
    nvrtcSuccess,
    
    -- NVRTC functions
    nvrtcGetErrorString,
    nvrtcCreateProgram,
    nvrtcCompileProgram,
    nvrtcGetProgramLogSize,
    nvrtcGetProgramLog,
    nvrtcGetPTXSize,
    nvrtcGetPTX,
    nvrtcDestroyProgram,
    
    -- CUDA Driver API functions
    cuInit,
    cuDeviceGet,
    cuCtxCreate,
    cuCtxDestroy,
    cuDevicePrimaryCtxRetain,
    cuDevicePrimaryCtxRelease,
    cuCtxSetCurrent,
    cuModuleLoadData,
    cuModuleGetFunction,
    cuMemAlloc,
    cuMemFree,
    cuMemcpyHtoD,
    cuMemcpyDtoH,
    cuLaunchKernel,
    cuCtxSynchronize,
    cuGetErrorString,
    cuModuleUnload
) where

import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.Ptr

-- CUDA Runtime API
foreign import ccall "cudaMalloc" cudaMalloc :: Ptr (Ptr ()) -> CSize -> IO CInt

foreign import ccall "cudaMemcpy" cudaMemcpy :: Ptr () -> Ptr () -> CSize -> CInt -> IO CInt

foreign import ccall "&cudaFree" cudaFree :: FinalizerPtr a

-- CUDA memcpy types
cudaMemcpyHostToDevice :: CInt
cudaMemcpyHostToDevice = 1

cudaMemcpyDeviceToHost :: CInt
cudaMemcpyDeviceToHost = 2

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

foreign import ccall "cuDevicePrimaryCtxRetain" cuDevicePrimaryCtxRetain :: Ptr CUcontext -> CUdevice -> IO CUresult

foreign import ccall "cuDevicePrimaryCtxRelease" cuDevicePrimaryCtxRelease :: CUdevice -> IO CUresult

foreign import ccall "cuCtxSetCurrent" cuCtxSetCurrent :: CUcontext -> IO CUresult

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