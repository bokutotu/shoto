{-# LANGUAGE LambdaCase #-}

module Runtime.NVIDIA.Internal.CUDASpec (spec) where

import           Control.Monad.IO.Class  (liftIO)
import qualified Data.ByteString         as BS
import           Foreign.C.Types         (CFloat)
import           Foreign.Marshal.Alloc   (free)
import           Foreign.Marshal.Array   (mallocArray, newArray, peekArray)
import           Foreign.Storable        (sizeOf)
import           Runtime.NVIDIA.Internal
import           Test.Hspec

spec :: Spec
spec = do
    describe "Runtime.NVIDIA.Internal" $ do
        it "initializes a context and queries compute capability" $ do
            result <- runCUDA computeCapability
            result `shouldSatisfy` \case
                Right (major, minor) -> major > 0 && minor >= 0
                Left _ -> False

        it "compiles a CUDA kernel to PTX with NVRTC" $ do
            result <- runCUDA $ do
                (major, minor) <- computeCapability
                compileProgramToPtx
                    "shoto_test.cu"
                    copyKernel
                    [ "--gpu-architecture=compute_" <> show major <> show minor
                    , "--std=c++11"
                    ]

            case result of
                Left err ->
                    expectationFailure $
                        "expected NVRTC compilation to succeed, but got: " <> show err
                Right ptx ->
                    BS.null ptx `shouldBe` False

        it "copies a host buffer to device and back" $ do
            let inputValues = [1, 2, 3, 4 :: CFloat]
                byteCount = length inputValues * sizeOf (undefined :: CFloat)

            result <- runCUDA $ do
                devicePtr <- allocBytes byteCount
                inputPtr <- liftIO $ newArray inputValues
                outputPtr <- liftIO $ mallocArray (length inputValues)
                copyBytesToDevice inputPtr byteCount devicePtr
                copyBytesFromDevice devicePtr byteCount outputPtr
                outputValues <- liftIO $ peekArray (length inputValues) outputPtr
                freeDevicePtr devicePtr
                liftIO $ free inputPtr
                liftIO $ free outputPtr
                pure outputValues

            result `shouldBe` Right inputValues

copyKernel :: String
copyKernel =
    unlines
        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B) {"
        , "    int i = blockIdx.x * blockDim.x + threadIdx.x;"
        , "    if (i < N) {"
        , "        A[i] = B[i];"
        , "    }"
        , "}"
        ]
