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
    -- Executor functions moved from Executor module
    compileGraph,
    executeGraph,
    CompiledKernel,
    CompiledGraph,
) where

import           Control.Monad          (unless, when)
import           Control.Monad.Cont     (ContT (ContT, runContT))
import qualified Data.ByteString.Char8  as BS
import           Data.ByteString.Unsafe (unsafeUseAsCString)
import           Data.IORef
import qualified Data.Map               as M
import           Foreign                (Storable (..), alloca, allocaArray,
                                         allocaBytes, malloc, peekArray, poke,
                                         withArray)
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.ForeignPtr     (ForeignPtr, castForeignPtr,
                                         newForeignPtr, withForeignPtr)
import           Foreign.Ptr
import           Internal.FFI
import           IR                     (Node (..), ValueId (..))
import           System.IO.Unsafe       (unsafePerformIO)
import qualified TinyIR                 as TIR

data GpuPtr a = GpuPtr {ptr :: ForeignPtr a, size :: Int}

withGpuPointers :: (Traversable t) => t (GpuPtr a) -> (t (Ptr ()) -> IO b) -> IO b
withGpuPointers containers action = runContT (traverse convert containers) action
  where
    convert gpu = ContT $ \cont -> withForeignPtr (ptr gpu) (cont . castPtr)

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

-- =============================
-- Executor functionality moved from Executor.hs
-- =============================

-- モジュールとファンクションのペアを保持
type CompiledKernel = (CUmodule, CUfunction)

type CompiledGraph = M.Map ValueId CompiledKernel

-- 現状は一つのOperationと2つか1つのInputがあるGraphのみを対称にする
-- TODO: 複数のOpsetへの対応
compileGraph :: TIR.TinyIR -> [ValueId] -> IO CompiledGraph
compileGraph ir outputIds = do
    initializeCuda
    case outputIds of
        [] -> pure M.empty
        (outId : _) ->
            case M.lookup outId ir of
                Just (Operation op inputs) -> do
                    -- CUDAコード生成
                    let cudaCode = generateCudaCode op inputs ir

                    -- コンパイルしてモジュールとファンクションを永続的に保持
                    withCudaContext 0 $ \_ -> do
                        -- PTXにコンパイル
                        ptx <- compileCudaKernel cudaCode (BS.pack "generated.cu") []

                        -- モジュールをロード（永続的に保持）
                        alloca $ \modulePtr -> do
                            BS.useAsCString ptx $ \ptxCStr -> do
                                checkCuda "cuModuleLoadData" =<< cuModuleLoadData modulePtr (castPtr ptxCStr)
                            modul <- peek modulePtr

                            -- ファンクションを取得
                            alloca $ \funcPtr -> do
                                BS.useAsCString (BS.pack "kernel") $ \funcName -> do
                                    checkCuda "cuModuleGetFunction" =<< cuModuleGetFunction funcPtr modul funcName
                                func <- peek funcPtr

                                return $ M.singleton outId (modul, func)
                _ -> return M.empty

-- 演算に応じたCUDAコード生成
generateCudaCode :: TIR.TinyOp -> [ValueId] -> TIR.TinyIR -> BS.ByteString
generateCudaCode (TIR.ElementWise op) _ _ =
    -- 入力のshapeを取得（今は固定で10と仮定）
    -- TODO: ちゃんと入力を使用するように修正
    TIR.codegenElementWise op []
generateCudaCode (TIR.Reduce op axis) _ _ =
    TIR.codegenReduce op axis []
generateCudaCode _ _ _ = error "Unsupported operation"

executeGraph ::
    forall a. TIR.TinyIR -> CompiledGraph -> [(ValueId, GpuPtr a)] -> (ValueId, GpuPtr a) -> IO ()
executeGraph ir functions inputs (outputId, output) = do
    -- CUDAコンテキストを設定
    withCudaContext 0 $ \_ -> do
        -- コンパイル済み関数を取得
        case M.lookup outputId functions of
            Nothing -> error "Function not compiled"
            Just (_, func) -> do
                -- モジュールとファンクションのペアを受け取る
                -- 演算の入力を特定
                case M.lookup outputId ir of
                    Just (Operation op opInputs) -> do
                        -- 入力ポインタを準備
                        let getPtr vid = lookup vid inputs
                            inputPtrs = mapM getPtr opInputs
                        case (op, inputPtrs) of
                            (TIR.ElementWise (TIR.Binary _), Just [input1, input2]) ->
                                -- Binary演算
                                launchBinaryKernel func input1 input2 output
                            (TIR.ElementWise (TIR.Unary _), Just [input1]) ->
                                -- Unary演算
                                launchUnaryKernel func input1 output
                            (TIR.Reduce _ Nothing, Just [input1]) ->
                                -- 全要素Reduce
                                launchReduceAllKernel func input1 output
                            (TIR.Reduce _ (Just axis), Just [input1]) ->
                                -- 軸指定Reduce
                                case M.lookup (ValueId 0) ir of
                                    Just (Input (TIR.Input shape)) ->
                                        launchReduceAxisKernel func input1 output axis shape
                                    _ -> error "Cannot determine input shape for axis reduction"
                            _ -> error $ "Unsupported operation/input configuration"
                    _ -> error "Output not found in IR"

-- Unary演算用のカーネル起動
launchUnaryKernel :: CUfunction -> GpuPtr a -> GpuPtr a -> IO ()
launchUnaryKernel func input output = do
    withGpuPointers [input, output] $ \[pIn, pOut] -> do
        let n = size output
            blockSize = 256
            gridSize = (n + blockSize - 1) `div` blockSize
            config =
                KernelLaunchConfig
                    { gridDimX = fromIntegral gridSize
                    , gridDimY = 1
                    , gridDimZ = 1
                    , blockDimX = fromIntegral blockSize
                    , blockDimY = 1
                    , blockDimZ = 1
                    , sharedMemBytes = 0
                    }
        -- カーネル引数の準備
        alloca $ \inPtr -> alloca $ \outPtr -> alloca $ \nPtr -> do
            poke inPtr pIn
            poke outPtr pOut
            poke nPtr (fromIntegral n :: CInt)
            let args = [castPtr inPtr, castPtr outPtr, castPtr nPtr]
            launchKernel func config args

-- 全要素Reduce用のカーネル起動
launchReduceAllKernel :: CUfunction -> GpuPtr a -> GpuPtr a -> IO ()
launchReduceAllKernel func input output = do
    withGpuPointers [input, output] $ \[pIn, pOut] -> do
        let n = size input
            -- 単一スレッドで実行（TODO: 並列化）
            config =
                KernelLaunchConfig
                    { gridDimX = 1
                    , gridDimY = 1
                    , gridDimZ = 1
                    , blockDimX = 1
                    , blockDimY = 1
                    , blockDimZ = 1
                    , sharedMemBytes = 0
                    }
        -- カーネル引数の準備
        alloca $ \inPtr -> alloca $ \outPtr -> alloca $ \nPtr -> do
            poke inPtr pIn
            poke outPtr pOut
            poke nPtr (fromIntegral n :: CInt)
            let args = [castPtr inPtr, castPtr outPtr, castPtr nPtr]
            launchKernel func config args

-- 軸指定Reduce用のカーネル起動
launchReduceAxisKernel :: CUfunction -> GpuPtr a -> GpuPtr a -> Int -> TIR.Shape -> IO ()
launchReduceAxisKernel func input output axis shape = do
    withGpuPointers [input, output] $ \[pIn, pOut] -> do
        let outputSize = size output
            reduceSize = case shape !! axis of
                TIR.Static n -> n
                _ -> error "Dynamic shape not supported"
            innerSize = product [n | (i, TIR.Static n) <- zip [0 ..] shape, i > axis]
            outerStride = reduceSize * innerSize
            blockSize = 256
            gridSize = (outputSize + blockSize - 1) `div` blockSize
            config =
                KernelLaunchConfig
                    { gridDimX = fromIntegral gridSize
                    , gridDimY = 1
                    , gridDimZ = 1
                    , blockDimX = fromIntegral blockSize
                    , blockDimY = 1
                    , blockDimZ = 1
                    , sharedMemBytes = 0
                    }
        -- カーネル引数の準備（6引数）
        alloca $ \inPtr -> alloca $ \outPtr ->
            alloca $ \n1Ptr -> alloca $ \n2Ptr -> alloca $ \n3Ptr -> alloca $ \n4Ptr -> do
                poke inPtr pIn
                poke outPtr pOut
                poke n1Ptr (fromIntegral outerStride :: CInt)
                poke n2Ptr (fromIntegral reduceSize :: CInt)
                poke n3Ptr (fromIntegral innerSize :: CInt)
                poke n4Ptr (fromIntegral outputSize :: CInt)
                let args = [castPtr inPtr, castPtr outPtr, castPtr n1Ptr, castPtr n2Ptr, castPtr n3Ptr, castPtr n4Ptr]
                launchKernel func config args

-- Binary演算用のカーネル起動
launchBinaryKernel :: CUfunction -> GpuPtr a -> GpuPtr a -> GpuPtr a -> IO ()
launchBinaryKernel func input1 input2 output = do
    withGpuPointers [input1, input2, output] $ \[p1, p2, pOut] -> do
        let n = size output
            blockSize = 256
            gridSize = (n + blockSize - 1) `div` blockSize
            config =
                KernelLaunchConfig
                    { gridDimX = fromIntegral gridSize
                    , gridDimY = 1
                    , gridDimZ = 1
                    , blockDimX = fromIntegral blockSize
                    , blockDimY = 1
                    , blockDimZ = 1
                    , sharedMemBytes = 0
                    }
        -- カーネル引数の準備
        -- CUDAカーネルの引数は、値へのポインタのポインタとして渡す必要がある
        alloca $ \aPtr -> alloca $ \bPtr -> alloca $ \cPtr -> alloca $ \nPtr -> do
            poke aPtr p1
            poke bPtr p2
            poke cPtr pOut
            poke nPtr (fromIntegral n :: CInt)
            let args = [castPtr aPtr, castPtr bPtr, castPtr cPtr, castPtr nPtr]
            launchKernel func config args
