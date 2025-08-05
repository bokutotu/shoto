{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Runtime (
    -- Original Runtime exports
    allocGpu,
    copyToGpu,
    copyToCpu,
    GpuPtr (..),
    -- NVRTC and JIT exports
    checkCuda,
    checkNvrtc,
    cudaInitialized,
    compileCudaKernel,
    withCudaContext,
    initializeCuda,
    withCompiledKernel,
    withCudaKernel,
    launchKernel,
    KernelLaunchConfig (..),
    -- Only necessary types for external use
    CUfunction,
) where

import           Control.Monad          (unless, when)
import qualified Data.ByteString.Char8  as BS
import           Data.ByteString.Unsafe (unsafeUseAsCString)
import           Data.IORef
import           Foreign                (Storable (peek, sizeOf), alloca,
                                         allocaArray, allocaBytes, malloc,
                                         peekArray, withArray)
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.Ptr
import           Internal.FFI
import           System.IO.Unsafe       (unsafePerformIO)

data GpuPtr a = GpuPtr {ptr :: ForeignPtr a, size :: Int}

allocGpu :: forall a. (Storable a) => Int -> IO (GpuPtr a)
allocGpu size = do
    -- Ensure CUDA is initialized for runtime API usage
    initializeCuda
    let bytes = fromIntegral $ size * sizeOf (undefined :: a)
    ptr <- malloc
    result <- cudaMalloc ptr bytes
    when (result /= 0) $ error $ "cudaMalloc Failed: " ++ show result
    devicePtr <- peek ptr
    fptr <- newForeignPtr cudaFree devicePtr
    pure $ GpuPtr (castForeignPtr fptr) size

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

-- JIT/NVRTC functionality
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

-- High-level compilation function
compileCudaKernel :: BS.ByteString -> BS.ByteString -> [BS.ByteString] -> IO BS.ByteString
compileCudaKernel kernelSource fileName options = do
    alloca $ \progPtr -> do
        unsafeUseAsCString kernelSource $ \kernelCStr ->
            BS.useAsCString fileName $ \nameCStr -> do
                result <- nvrtcCreateProgram progPtr kernelCStr nameCStr 0 nullPtr nullPtr
                checkNvrtc "nvrtcCreateProgram" result

                prog <- peek progPtr

                -- Convert options to C strings and compile
                withByteStringsCStrings options $ \optionsPtrs ->
                    withArray optionsPtrs $ \optionsPtr -> do
                        compileResult <- nvrtcCompileProgram prog (fromIntegral $ length options) optionsPtr

                        -- Get compilation log
                        alloca $ \logSizePtr -> do
                            _ <- nvrtcGetProgramLogSize prog logSizePtr
                            logSize <- peek logSizePtr

                            when (logSize > 1) $ do
                                allocaBytes (fromIntegral logSize) $ \logPtr -> do
                                    _ <- nvrtcGetProgramLog prog logPtr
                                    logStr <- peekCString logPtr
                                    putStrLn $ "Compilation log:\n" ++ logStr

                        checkNvrtc "nvrtcCompileProgram" compileResult

                -- Get PTX
                alloca $ \ptxSizePtr -> do
                    _ <- nvrtcGetPTXSize prog ptxSizePtr
                    ptxSize <- peek ptxSizePtr

                    allocaBytes (fromIntegral ptxSize) $ \ptxPtr -> do
                        _ <- nvrtcGetPTX prog ptxPtr
                        ptxBS <- BS.packCStringLen (ptxPtr, fromIntegral ptxSize - 1)

                        -- Cleanup
                        _ <- nvrtcDestroyProgram progPtr

                        return ptxBS

-- Helper to convert multiple ByteStrings to C strings
withByteStringsCStrings :: [BS.ByteString] -> ([CString] -> IO a) -> IO a
withByteStringsCStrings [] f = f []
withByteStringsCStrings (s : ss) f = BS.useAsCString s $ \cs ->
    withByteStringsCStrings ss $ \css -> f (cs : css)

-- Initialize CUDA (idempotent)
initializeCuda :: IO ()
initializeCuda = do
    alreadyInit <- readIORef cudaInitialized
    unless alreadyInit $ do
        checkCuda "cuInit" =<< cuInit 0
        writeIORef cudaInitialized True

-- High-level context management
withCudaContext :: Int -> (CUcontext -> IO a) -> IO a
withCudaContext deviceNum action = do
    initializeCuda
    alloca $ \devicePtr -> do
        checkCuda "cuDeviceGet" =<< cuDeviceGet devicePtr (fromIntegral deviceNum)
        device <- peek devicePtr

        alloca $ \ctxPtr -> do
            -- Use primary context to be compatible with Runtime API
            checkCuda "cuDevicePrimaryCtxRetain" =<< cuDevicePrimaryCtxRetain ctxPtr device
            ctx <- peek ctxPtr

            -- Set as current context
            checkCuda "cuCtxSetCurrent" =<< cuCtxSetCurrent ctx

            result <- action ctx

            checkCuda "cuDevicePrimaryCtxRelease" =<< cuDevicePrimaryCtxRelease device

            return result

-- High-level function to compile and use a kernel
withCompiledKernel ::
    BS.ByteString -> BS.ByteString -> [BS.ByteString] -> BS.ByteString -> (CUfunction -> IO a) -> IO a
withCompiledKernel kernelSource fileName compileOptions functionName action = do
    -- Compile kernel to PTX
    ptx <- compileCudaKernel kernelSource fileName compileOptions

    -- Load module
    alloca $ \modulePtr -> do
        BS.useAsCString ptx $ \ptxCStr -> do
            checkCuda "cuModuleLoadData" =<< cuModuleLoadData modulePtr (castPtr ptxCStr)

        modul <- peek modulePtr

        -- Get function
        alloca $ \funcPtr -> do
            BS.useAsCString functionName $ \funcName -> do
                checkCuda "cuModuleGetFunction" =<< cuModuleGetFunction funcPtr modul funcName

            func <- peek funcPtr

            -- Execute action with function
            result <- action func

            -- Cleanup
            checkCuda "cuModuleUnload" =<< cuModuleUnload modul

            return result

-- Kernel launch configuration
data KernelLaunchConfig = KernelLaunchConfig
    { gridDimX :: CUInt
    , gridDimY :: CUInt
    , gridDimZ :: CUInt
    , blockDimX :: CUInt
    , blockDimY :: CUInt
    , blockDimZ :: CUInt
    , sharedMemBytes :: CUInt
    }
    deriving (Show, Eq)

-- High-level kernel launch function
launchKernel :: CUfunction -> KernelLaunchConfig -> [Ptr ()] -> IO ()
launchKernel func config args = do
    withArray args $ \argsPtr -> do
        checkCuda "cuLaunchKernel"
            =<< cuLaunchKernel
                func
                (gridDimX config)
                (gridDimY config)
                (gridDimZ config)
                (blockDimX config)
                (blockDimY config)
                (blockDimZ config)
                (sharedMemBytes config)
                nullPtr -- default stream
                argsPtr
                nullPtr -- extra options
    checkCuda "cuCtxSynchronize" =<< cuCtxSynchronize

-- Complete high-level function that combines context, compilation, and execution
withCudaKernel ::
    Int -> -- Device number
    BS.ByteString -> -- Kernel source
    BS.ByteString -> -- File name
    [BS.ByteString] -> -- Compile options
    BS.ByteString -> -- Function name
    (CUfunction -> IO a) -> -- Action to run with the compiled function
    IO a
withCudaKernel deviceNum kernelSource fileName compileOptions functionName action = do
    withCudaContext deviceNum $ \_ -> do
        withCompiledKernel kernelSource fileName compileOptions functionName action
