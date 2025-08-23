{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE StandaloneDeriving #-}

module Internal.FFI (
    -- CUDA Runtime API
    cudaMalloc,
    cudaMemcpy,
    cudaFree,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    
    -- NVRTC Types
    NvrtcProgram(..),
    NvrtcResult(..),
    
    -- CUDA Driver API Types
    CUdevice(..),
    CUcontext(..),
    CUmodule(..),
    CUfunction(..),
    CUdeviceptr(..),
    CUresult(..),
    CUstream(..),
    
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
import           Foreign.Storable

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
newtype NvrtcProgram = NvrtcProgram (Ptr ())
    deriving (Eq, Show, Storable)

newtype NvrtcResult = NvrtcResult CInt
    deriving (Eq, Show, Num, Storable)

-- CUDA Driver API Types
newtype CUdevice = CUdevice CInt
    deriving (Eq, Show, Num, Storable)

newtype CUcontext = CUcontext (Ptr ())
    deriving (Eq, Show, Storable)

newtype CUmodule = CUmodule (Ptr ())
    deriving (Eq, Show, Storable)

newtype CUfunction = CUfunction (Ptr ())
    deriving (Eq, Show, Storable)

newtype CUdeviceptr = CUdeviceptr CULLong
    deriving (Eq, Show, Num, Enum, Ord, Real, Integral, Storable)

newtype CUresult = CUresult CInt
    deriving (Eq, Show, Num, Storable)

newtype CUstream = CUstream (Ptr ())
    deriving (Eq, Show, Storable)

-- Result codes
cudaSuccess :: CInt
cudaSuccess = 0

nvrtcSuccess :: NvrtcResult
nvrtcSuccess = NvrtcResult 0

-- NVRTC FFI declarations
foreign import ccall "nvrtcGetErrorString" nvrtcGetErrorString' :: CInt -> IO CString

nvrtcGetErrorString :: NvrtcResult -> IO CString
nvrtcGetErrorString (NvrtcResult res) = nvrtcGetErrorString' res

foreign import ccall "nvrtcCreateProgram"
    nvrtcCreateProgram' ::
        Ptr (Ptr ()) -> CString -> CString -> CInt -> Ptr CString -> Ptr CString -> IO CInt

nvrtcCreateProgram :: Ptr NvrtcProgram -> CString -> CString -> CInt -> Ptr CString -> Ptr CString -> IO NvrtcResult
nvrtcCreateProgram pProg src name nHeaders headers headerNames = 
    NvrtcResult <$> nvrtcCreateProgram' (castPtr pProg) src name nHeaders headers headerNames

foreign import ccall "nvrtcCompileProgram"
    nvrtcCompileProgram' ::
        Ptr () -> CInt -> Ptr CString -> IO CInt

nvrtcCompileProgram :: NvrtcProgram -> CInt -> Ptr CString -> IO NvrtcResult
nvrtcCompileProgram (NvrtcProgram prog) nOpts opts = 
    NvrtcResult <$> nvrtcCompileProgram' prog nOpts opts

foreign import ccall "nvrtcGetProgramLogSize"
    nvrtcGetProgramLogSize' ::
        Ptr () -> Ptr CSize -> IO CInt

nvrtcGetProgramLogSize :: NvrtcProgram -> Ptr CSize -> IO NvrtcResult
nvrtcGetProgramLogSize (NvrtcProgram prog) pLogSize = 
    NvrtcResult <$> nvrtcGetProgramLogSize' prog pLogSize

foreign import ccall "nvrtcGetProgramLog"
    nvrtcGetProgramLog' ::
        Ptr () -> CString -> IO CInt

nvrtcGetProgramLog :: NvrtcProgram -> CString -> IO NvrtcResult
nvrtcGetProgramLog (NvrtcProgram prog) logBuf = 
    NvrtcResult <$> nvrtcGetProgramLog' prog logBuf

foreign import ccall "nvrtcGetPTXSize"
    nvrtcGetPTXSize' ::
        Ptr () -> Ptr CSize -> IO CInt

nvrtcGetPTXSize :: NvrtcProgram -> Ptr CSize -> IO NvrtcResult
nvrtcGetPTXSize (NvrtcProgram prog) pPtxSize = 
    NvrtcResult <$> nvrtcGetPTXSize' prog pPtxSize

foreign import ccall "nvrtcGetPTX"
    nvrtcGetPTX' ::
        Ptr () -> CString -> IO CInt

nvrtcGetPTX :: NvrtcProgram -> CString -> IO NvrtcResult
nvrtcGetPTX (NvrtcProgram prog) ptxBuf = 
    NvrtcResult <$> nvrtcGetPTX' prog ptxBuf

foreign import ccall "nvrtcDestroyProgram"
    nvrtcDestroyProgram' ::
        Ptr (Ptr ()) -> IO CInt

nvrtcDestroyProgram :: Ptr NvrtcProgram -> IO NvrtcResult
nvrtcDestroyProgram pProg = 
    NvrtcResult <$> nvrtcDestroyProgram' (castPtr pProg)

-- CUDA Driver API FFI declarations
foreign import ccall "cuInit" cuInit' :: CUInt -> IO CInt

cuInit :: CUInt -> IO CUresult
cuInit flags = CUresult <$> cuInit' flags

foreign import ccall "cuDeviceGet" cuDeviceGet' :: Ptr CInt -> CInt -> IO CInt

cuDeviceGet :: Ptr CUdevice -> CInt -> IO CUresult
cuDeviceGet pDev ordinal = CUresult <$> cuDeviceGet' (castPtr pDev) ordinal

foreign import ccall "cuCtxCreate" cuCtxCreate' :: Ptr (Ptr ()) -> CUInt -> CInt -> IO CInt

cuCtxCreate :: Ptr CUcontext -> CUInt -> CUdevice -> IO CUresult
cuCtxCreate pCtx flags (CUdevice dev) = CUresult <$> cuCtxCreate' (castPtr pCtx) flags dev

foreign import ccall "cuCtxDestroy" cuCtxDestroy' :: Ptr () -> IO CInt

cuCtxDestroy :: CUcontext -> IO CUresult
cuCtxDestroy (CUcontext ctx) = CUresult <$> cuCtxDestroy' ctx

foreign import ccall "cuDevicePrimaryCtxRetain" cuDevicePrimaryCtxRetain' :: Ptr (Ptr ()) -> CInt -> IO CInt

cuDevicePrimaryCtxRetain :: Ptr CUcontext -> CUdevice -> IO CUresult
cuDevicePrimaryCtxRetain pCtx (CUdevice dev) = CUresult <$> cuDevicePrimaryCtxRetain' (castPtr pCtx) dev

foreign import ccall "cuDevicePrimaryCtxRelease" cuDevicePrimaryCtxRelease' :: CInt -> IO CInt

cuDevicePrimaryCtxRelease :: CUdevice -> IO CUresult
cuDevicePrimaryCtxRelease (CUdevice dev) = CUresult <$> cuDevicePrimaryCtxRelease' dev

foreign import ccall "cuCtxSetCurrent" cuCtxSetCurrent' :: Ptr () -> IO CInt

cuCtxSetCurrent :: CUcontext -> IO CUresult
cuCtxSetCurrent (CUcontext ctx) = CUresult <$> cuCtxSetCurrent' ctx

foreign import ccall "cuModuleLoadData" cuModuleLoadData' :: Ptr (Ptr ()) -> Ptr () -> IO CInt

cuModuleLoadData :: Ptr CUmodule -> Ptr () -> IO CUresult
cuModuleLoadData pMod pImage = CUresult <$> cuModuleLoadData' (castPtr pMod) pImage

foreign import ccall "cuModuleGetFunction"
    cuModuleGetFunction' :: Ptr (Ptr ()) -> Ptr () -> CString -> IO CInt

cuModuleGetFunction :: Ptr CUfunction -> CUmodule -> CString -> IO CUresult
cuModuleGetFunction pFunc (CUmodule hmod) name = 
    CUresult <$> cuModuleGetFunction' (castPtr pFunc) hmod name

foreign import ccall "cuMemAlloc" cuMemAlloc' :: Ptr CULLong -> CSize -> IO CInt

cuMemAlloc :: Ptr CUdeviceptr -> CSize -> IO CUresult
cuMemAlloc pDptr size = CUresult <$> cuMemAlloc' (castPtr pDptr) size

foreign import ccall "cuMemFree" cuMemFree' :: CULLong -> IO CInt

cuMemFree :: CUdeviceptr -> IO CUresult
cuMemFree (CUdeviceptr dptr) = CUresult <$> cuMemFree' dptr

foreign import ccall "cuMemcpyHtoD" cuMemcpyHtoD' :: CULLong -> Ptr () -> CSize -> IO CInt

cuMemcpyHtoD :: CUdeviceptr -> Ptr () -> CSize -> IO CUresult
cuMemcpyHtoD (CUdeviceptr dptr) pSrc size = CUresult <$> cuMemcpyHtoD' dptr pSrc size

foreign import ccall "cuMemcpyDtoH" cuMemcpyDtoH' :: Ptr () -> CULLong -> CSize -> IO CInt

cuMemcpyDtoH :: Ptr () -> CUdeviceptr -> CSize -> IO CUresult
cuMemcpyDtoH pDst (CUdeviceptr dptr) size = CUresult <$> cuMemcpyDtoH' pDst dptr size

foreign import ccall "cuLaunchKernel"
    cuLaunchKernel' ::
        Ptr () ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        Ptr () ->
        Ptr (Ptr ()) ->
        Ptr (Ptr ()) ->
        IO CInt

cuLaunchKernel :: CUfunction -> CUInt -> CUInt -> CUInt -> CUInt -> CUInt -> CUInt -> CUInt -> CUstream -> Ptr (Ptr ()) -> Ptr (Ptr ()) -> IO CUresult
cuLaunchKernel (CUfunction f) gx gy gz bx by bz sm (CUstream strm) params extra = 
    CUresult <$> cuLaunchKernel' f gx gy gz bx by bz sm strm params extra

foreign import ccall "cuCtxSynchronize" cuCtxSynchronize' :: IO CInt

cuCtxSynchronize :: IO CUresult
cuCtxSynchronize = CUresult <$> cuCtxSynchronize'

foreign import ccall "cuGetErrorString" cuGetErrorString' :: CInt -> Ptr CString -> IO CInt

cuGetErrorString :: CUresult -> Ptr CString -> IO CUresult
cuGetErrorString (CUresult res) pStr = CUresult <$> cuGetErrorString' res pStr

foreign import ccall "cuModuleUnload" cuModuleUnload' :: Ptr () -> IO CInt

cuModuleUnload :: CUmodule -> IO CUresult
cuModuleUnload (CUmodule hmod) = CUresult <$> cuModuleUnload' hmod