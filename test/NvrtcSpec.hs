module NvrtcSpec where

import           Control.Monad          (when)
import qualified Data.ByteString.Char8  as BS
import           Data.ByteString.Unsafe (unsafeUseAsCString)
import           Foreign
import           Foreign.C.String
import           Foreign.C.Types
import           Nvrtc
import           System.Exit            (exitFailure)
import           Test.Hspec

-- Vector addition kernel
vectorAddKernel :: BS.ByteString
vectorAddKernel =
    BS.pack $
        Prelude.unlines
            [ "extern \"C\" __global__ void vector_add(float* a, float* b, float* c, int n) {"
            , "    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
            , "    if (idx < n) {"
            , "        c[idx] = a[idx] + b[idx];"
            , "    }"
            , "}"
            ]

spec :: Spec
spec = describe "Nvrtc JIT test" $ do
    it "add kernel" $ do
        -- Initialize CUDA
        checkCuda "cuInit" =<< cuInit 0

        -- Get device and create context
        alloca $ \devicePtr -> do
            checkCuda "cuDeviceGet" =<< cuDeviceGet devicePtr 0
            device <- peek devicePtr

            alloca $ \ctxPtr -> do
                checkCuda "cuCtxCreate" =<< cuCtxCreate ctxPtr 0 device
                ctx <- peek ctxPtr

                -- Create and compile NVRTC program
                ptx <- compileKernel

                -- Load module and get function
                alloca $ \modulePtr -> do
                    BS.useAsCString ptx $ \ptxCStr -> do
                        checkCuda "cuModuleLoadData" =<< cuModuleLoadData modulePtr (castPtr ptxCStr)

                    modul <- peek modulePtr

                    alloca $ \funcPtr -> do
                        withCString "vector_add" $ \funcName -> do
                            checkCuda "cuModuleGetFunction" =<< cuModuleGetFunction funcPtr modul funcName

                        func <- peek funcPtr

                        -- Run vector addition test
                        runVectorAddition func

                        -- Cleanup
                        checkCuda "cuModuleUnload" =<< cuModuleUnload modul

                checkCuda "cuCtxDestroy" =<< cuCtxDestroy ctx

        Prelude.putStrLn "\n✓ All tests passed!"

compileKernel :: IO BS.ByteString
compileKernel = do
    alloca $ \progPtr -> do
        unsafeUseAsCString vectorAddKernel $ \kernelCStr ->
            withCString "vector_add.cu" $ \nameCStr -> do
                result <- nvrtcCreateProgram progPtr kernelCStr nameCStr 0 nullPtr nullPtr
                checkNvrtc "nvrtcCreateProgram" result

                prog <- peek progPtr

                -- Compile
                withCString "--gpu-architecture=compute_70" $ \optionCStr ->
                    alloca $ \optionsPtr -> do
                        poke optionsPtr optionCStr
                        compileResult <- nvrtcCompileProgram prog 1 optionsPtr

                        -- Get log
                        alloca $ \logSizePtr -> do
                            _ <- nvrtcGetProgramLogSize prog logSizePtr
                            logSize <- peek logSizePtr

                            when (logSize > 1) $ do
                                allocaBytes (fromIntegral logSize) $ \logPtr -> do
                                    _ <- nvrtcGetProgramLog prog logPtr
                                    log <- peekCString logPtr
                                    Prelude.putStrLn $ "Compilation log:\n" ++ log

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

runVectorAddition :: CUfunction -> IO ()
runVectorAddition func = do
    let n = 1024 :: Int
        bytes = fromIntegral $ n * sizeOf (undefined :: CFloat)

    -- Allocate device memory
    alloca $ \daPtr -> alloca $ \dbPtr -> alloca $ \dcPtr -> do
        checkCuda "cuMemAlloc a" =<< cuMemAlloc daPtr bytes
        checkCuda "cuMemAlloc b" =<< cuMemAlloc dbPtr bytes
        checkCuda "cuMemAlloc c" =<< cuMemAlloc dcPtr bytes

        da <- peek daPtr
        db <- peek dbPtr
        dc <- peek dcPtr

        -- Prepare host data
        let ha = Prelude.replicate n (1.0 :: CFloat)
            hb = Prelude.replicate n (2.0 :: CFloat)

        -- Copy to device
        withArray ha $ \haPtr -> checkCuda "cuMemcpyHtoD a" =<< cuMemcpyHtoD da (castPtr haPtr) bytes
        withArray hb $ \hbPtr -> checkCuda "cuMemcpyHtoD b" =<< cuMemcpyHtoD db (castPtr hbPtr) bytes

        -- Launch kernel
        let threadsPerBlock = 256 :: CUInt
            blocksPerGrid = fromIntegral $ (n + fromIntegral threadsPerBlock - 1) `div` fromIntegral threadsPerBlock

        alloca $ \nPtr -> do
            poke nPtr (fromIntegral n :: CInt)

            withArray [castPtr daPtr, castPtr dbPtr, castPtr dcPtr, castPtr nPtr] $ \argsPtr -> do
                checkCuda "cuLaunchKernel"
                    =<< cuLaunchKernel
                        func
                        blocksPerGrid
                        1
                        1
                        threadsPerBlock
                        1
                        1
                        0
                        nullPtr
                        argsPtr
                        nullPtr

        -- Synchronize
        checkCuda "cuCtxSynchronize" =<< cuCtxSynchronize

        -- Copy result back
        allocaArray n $ \hcPtr -> do
            checkCuda "cuMemcpyDtoH" =<< cuMemcpyDtoH (castPtr hcPtr) dc bytes
            hc <- peekArray n hcPtr

            -- Verify
            let expected = 3.0 :: CFloat
                errors = [(i, v) | (i, v) <- Prelude.zip [0 ..] hc, abs (v - expected) > 0.0001]

            if Prelude.null errors
                then Prelude.putStrLn "✓ Vector addition successful!"
                else do
                    Prelude.putStrLn "✗ Vector addition failed:"
                    mapM_
                        (\(i, v) -> Prelude.putStrLn $ "  Index " ++ show i ++ ": expected 3.0, got " ++ show v)
                        (Prelude.take 5 errors)
                    exitFailure

        -- Free device memory
        checkCuda "cuMemFree a" =<< cuMemFree da
        checkCuda "cuMemFree b" =<< cuMemFree db
        checkCuda "cuMemFree c" =<< cuMemFree dc
