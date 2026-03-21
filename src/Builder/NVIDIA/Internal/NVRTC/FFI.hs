{-# LANGUAGE ForeignFunctionInterface #-}

module Builder.NVIDIA.Internal.NVRTC.FFI (
    NvrtcResult,
    RawProgram,
    nvrtcSuccess,
    c_nvrtcVersion,
    c_nvrtcCreateProgram,
    c_nvrtcDestroyProgram,
    c_nvrtcCompileProgram,
    c_nvrtcGetProgramLogSize,
    c_nvrtcGetProgramLog,
    c_nvrtcGetPTXSize,
    c_nvrtcGetPTX,
    c_nvrtcGetErrorString,
) where

import           Foreign.C.String (CString)
import           Foreign.C.Types  (CInt (..), CSize (..))
import           Foreign.Ptr      (Ptr)

type NvrtcResult = CInt

data NvrtcProgram

type RawProgram = Ptr NvrtcProgram

nvrtcSuccess :: NvrtcResult
nvrtcSuccess = 0

foreign import ccall unsafe "nvrtcVersion"
    c_nvrtcVersion :: Ptr CInt -> Ptr CInt -> IO NvrtcResult

foreign import ccall unsafe "nvrtcCreateProgram"
    c_nvrtcCreateProgram ::
        Ptr RawProgram ->
        CString ->
        CString ->
        CInt ->
        Ptr CString ->
        Ptr CString ->
        IO NvrtcResult

foreign import ccall unsafe "nvrtcDestroyProgram"
    c_nvrtcDestroyProgram :: Ptr RawProgram -> IO NvrtcResult

foreign import ccall unsafe "nvrtcCompileProgram"
    c_nvrtcCompileProgram :: RawProgram -> CInt -> Ptr CString -> IO NvrtcResult

foreign import ccall unsafe "nvrtcGetProgramLogSize"
    c_nvrtcGetProgramLogSize :: RawProgram -> Ptr CSize -> IO NvrtcResult

foreign import ccall unsafe "nvrtcGetProgramLog"
    c_nvrtcGetProgramLog :: RawProgram -> CString -> IO NvrtcResult

foreign import ccall unsafe "nvrtcGetPTXSize"
    c_nvrtcGetPTXSize :: RawProgram -> Ptr CSize -> IO NvrtcResult

foreign import ccall unsafe "nvrtcGetPTX"
    c_nvrtcGetPTX :: RawProgram -> CString -> IO NvrtcResult

foreign import ccall unsafe "nvrtcGetErrorString"
    c_nvrtcGetErrorString :: NvrtcResult -> IO CString
