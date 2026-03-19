{-# LANGUAGE ForeignFunctionInterface #-}

module Runtime.NVIDIA.Internal.Driver.FFI (
    CuResult,
    CuDevice,
    CuDevicePtr,
    RawContext,
    RawModule,
    RawFunction,
    RawStream,
    CStringResultPtr,
    cuSuccess,
    cuDeviceAttributeComputeCapabilityMajor,
    cuDeviceAttributeComputeCapabilityMinor,
    c_cuInit,
    c_cuDeviceGet,
    c_cuDeviceGetAttribute,
    c_cuCtxCreate,
    c_cuCtxDestroy,
    c_cuModuleLoadData,
    c_cuModuleUnload,
    c_cuModuleGetFunction,
    c_cuMemAlloc,
    c_cuMemFree,
    c_cuMemcpyHtoD,
    c_cuMemcpyDtoH,
    c_cuLaunchKernel,
    c_cuCtxSynchronize,
    c_cuGetErrorName,
    c_cuGetErrorString,
) where

import           Data.Word        (Word64)
import           Foreign.C.String (CString)
import           Foreign.C.Types  (CInt (..), CSize (..), CUInt (..))
import           Foreign.Ptr      (Ptr)

type CuResult = CInt

type CuDevice = CInt

type CuDevicePtr = Word64

data CuContext

type RawContext = Ptr CuContext

data CuModule

type RawModule = Ptr CuModule

data CuFunction

type RawFunction = Ptr CuFunction

data CuStream

type RawStream = Ptr CuStream

type CStringResultPtr = Ptr CString

cuSuccess :: CuResult
cuSuccess = 0

cuDeviceAttributeComputeCapabilityMajor :: CInt
cuDeviceAttributeComputeCapabilityMajor = 75

cuDeviceAttributeComputeCapabilityMinor :: CInt
cuDeviceAttributeComputeCapabilityMinor = 76

foreign import ccall unsafe "cuInit"
    c_cuInit :: CUInt -> IO CuResult

foreign import ccall unsafe "cuDeviceGet"
    c_cuDeviceGet :: Ptr CuDevice -> CInt -> IO CuResult

foreign import ccall unsafe "cuDeviceGetAttribute"
    c_cuDeviceGetAttribute :: Ptr CInt -> CInt -> CuDevice -> IO CuResult

foreign import ccall unsafe "cuCtxCreate_v2"
    c_cuCtxCreate :: Ptr RawContext -> CUInt -> CuDevice -> IO CuResult

foreign import ccall unsafe "cuCtxDestroy_v2"
    c_cuCtxDestroy :: RawContext -> IO CuResult

foreign import ccall unsafe "cuModuleLoadData"
    c_cuModuleLoadData :: Ptr RawModule -> Ptr () -> IO CuResult

foreign import ccall unsafe "cuModuleUnload"
    c_cuModuleUnload :: RawModule -> IO CuResult

foreign import ccall unsafe "cuModuleGetFunction"
    c_cuModuleGetFunction :: Ptr RawFunction -> RawModule -> CString -> IO CuResult

foreign import ccall unsafe "cuMemAlloc_v2"
    c_cuMemAlloc :: Ptr CuDevicePtr -> CSize -> IO CuResult

foreign import ccall unsafe "cuMemFree_v2"
    c_cuMemFree :: CuDevicePtr -> IO CuResult

foreign import ccall unsafe "cuMemcpyHtoD_v2"
    c_cuMemcpyHtoD :: CuDevicePtr -> Ptr a -> CSize -> IO CuResult

foreign import ccall unsafe "cuMemcpyDtoH_v2"
    c_cuMemcpyDtoH :: Ptr a -> CuDevicePtr -> CSize -> IO CuResult

foreign import ccall unsafe "cuLaunchKernel"
    c_cuLaunchKernel ::
        RawFunction ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        CUInt ->
        RawStream ->
        Ptr (Ptr ()) ->
        Ptr () ->
        IO CuResult

foreign import ccall unsafe "cuCtxSynchronize"
    c_cuCtxSynchronize :: IO CuResult

foreign import ccall unsafe "cuGetErrorName"
    c_cuGetErrorName :: CuResult -> CStringResultPtr -> IO CuResult

foreign import ccall unsafe "cuGetErrorString"
    c_cuGetErrorString :: CuResult -> CStringResultPtr -> IO CuResult
