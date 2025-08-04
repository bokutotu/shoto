{-# LANGUAGE TypeApplications #-}

module NvrtcSpec where

import qualified Data.ByteString.Char8  as BS
import           Foreign
import           Foreign.C.Types
import           Nvrtc (CUfunction)
import           NvrtcWrapper
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
        let compileOptions = 
                [ "--gpu-architecture=compute_70"
                , "-default-device"        -- デフォルトデバイス設定
                , "--use_fast_math"        -- 高速数学関数
                , "--fmad=true"            -- FMA命令の使用
                ]
        withCudaKernel 0 vectorAddKernel (BS.pack "vector_add.cu") (map BS.pack compileOptions) (BS.pack "vector_add") $ \func -> do
            runVectorAddition func

        Prelude.putStrLn "\n✓ All tests passed!"

runVectorAddition :: CUfunction -> IO ()
runVectorAddition func = do
    let n = 1024 :: Int
        
    -- Prepare host data
    let ha = Prelude.replicate n (1.0 :: CFloat)
        hb = Prelude.replicate n (2.0 :: CFloat)

    -- Use device memory with automatic cleanup
    withDeviceArray @CFloat n $ \da -> do
        withDeviceArray @CFloat n $ \db -> do
            withDeviceArray @CFloat n $ \dc -> do
                -- Copy input data to device
                copyToDevice da ha
                copyToDevice db hb

                -- Launch kernel
                let threadsPerBlock = 256 :: CUInt
                    blocksPerGrid = fromIntegral $ (n + fromIntegral threadsPerBlock - 1) `div` fromIntegral threadsPerBlock
                    config = KernelLaunchConfig 
                        { gridDimX = blocksPerGrid
                        , gridDimY = 1
                        , gridDimZ = 1
                        , blockDimX = threadsPerBlock
                        , blockDimY = 1
                        , blockDimZ = 1
                        , sharedMemBytes = 0
                        }

                alloca $ \daPtr -> alloca $ \dbPtr -> alloca $ \dcPtr -> alloca $ \nPtr -> do
                    poke daPtr da
                    poke dbPtr db
                    poke dcPtr dc
                    poke nPtr (fromIntegral n :: CInt)

                    launchKernel func config [castPtr daPtr, castPtr dbPtr, castPtr dcPtr, castPtr nPtr]

                -- Copy result back and verify
                hc <- copyFromDevice n dc

                -- Verify
                let expected = 3.0 :: CFloat
                    errors = [(i, v) | (i, v) <- Prelude.zip [0 :: Int ..] hc, abs (v - expected) > 0.0001]

                if Prelude.null errors
                    then Prelude.putStrLn "✓ Vector addition successful!"
                    else do
                        Prelude.putStrLn "✗ Vector addition failed:"
                        mapM_
                            (\(i, v) -> Prelude.putStrLn $ "  Index " ++ show i ++ ": expected 3.0, got " ++ show v)
                            (Prelude.take 5 errors)
                        exitFailure
