{-# LANGUAGE LambdaCase #-}

module Runtime.NVIDIASpec (spec) where

import qualified Builder.NVIDIA as BuilderNVIDIA
import           Builder.Types  (BuilderError (..))
import           Codegen        (CudaDim (..))
import           Runtime.NVIDIA (allocateDeviceBuffer, downloadTensorBuffer,
                                 freeDeviceBuffer, loadNvidiaKernel, runNVIDIA,
                                 runNvidiaKernel,
                                 runNvidiaKernelWithHostBuffers,
                                 uploadTensorBuffer)
import           Runtime.Types  (KernelArg (..), KernelSignature (..),
                                 RuntimeError (..), emptyTensorBuffer,
                                 readTensorBuffer, tensorBufferFromList)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Runtime.NVIDIA" $ do
        it "compiles and runs a copy kernel with host-buffer staging" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]
            compiledKernel <- expectCompiledCudaProgram copyKernelSignature CudaX copyKernel

            actual <- runNVIDIA $ do
                loadedKernel <- loadNvidiaKernel compiledKernel
                runNvidiaKernelWithHostBuffers
                    loadedKernel
                    64
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Right ()
            readTensorBuffer output `shouldReturn` [1, 2, 3, 4]

        it "runs a pointwise add kernel with explicit device buffers" $ do
            output <- emptyTensorBuffer 4
            lhs <- tensorBufferFromList [1, 2, 3, 4]
            rhs <- tensorBufferFromList [5, 6, 7, 8]
            compiledKernel <- expectCompiledCudaProgram addKernelSignature CudaX addKernel

            actual <- runNVIDIA $ do
                loadedKernel <- loadNvidiaKernel compiledKernel
                outputBuffer <- allocateDeviceBuffer 4
                lhsBuffer <- allocateDeviceBuffer 4
                rhsBuffer <- allocateDeviceBuffer 4
                uploadTensorBuffer lhs lhsBuffer
                uploadTensorBuffer rhs rhsBuffer
                runNvidiaKernel loadedKernel 128 4 [outputBuffer, lhsBuffer, rhsBuffer]
                downloadTensorBuffer outputBuffer output
                freeDeviceBuffer outputBuffer
                freeDeviceBuffer lhsBuffer
                freeDeviceBuffer rhsBuffer

            actual `shouldBe` Right ()
            readTensorBuffer output `shouldReturn` [6, 8, 10, 12]

        it "reports NVRTC compilation failures" $ do
            BuilderNVIDIA.compileCudaProgram copyKernelSignature CudaX malformedKernel
                >>= (`shouldSatisfy` isNvrtcFailure)

        it "rejects invalid thread block sizes before launch" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]
            compiledKernel <- expectCompiledCudaProgram copyKernelSignature CudaX copyKernel

            actual <- runNVIDIA $ do
                loadedKernel <- loadNvidiaKernel compiledKernel
                runNvidiaKernelWithHostBuffers
                    loadedKernel
                    0
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Left (ErrRuntimeInvalidThreadBlockSize 0)

        it "rejects tensor buffers smaller than the requested extent" $ do
            output <- emptyTensorBuffer 2
            input <- tensorBufferFromList [1, 2]
            compiledKernel <- expectCompiledCudaProgram copyKernelSignature CudaX copyKernel

            actual <- runNVIDIA $ do
                loadedKernel <- loadNvidiaKernel compiledKernel
                runNvidiaKernelWithHostBuffers
                    loadedKernel
                    32
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Left (ErrRuntimeTensorTooSmall 1 4 2)

expectCompiledCudaProgram ::
    KernelSignature ->
    CudaDim ->
    String ->
    IO BuilderNVIDIA.CompiledCudaProgram
expectCompiledCudaProgram kernelSignature cudaDim source = do
    Right compiledProgram <- BuilderNVIDIA.compileCudaProgram kernelSignature cudaDim source
    pure compiledProgram

isNvrtcFailure :: Either BuilderError a -> Bool
isNvrtcFailure = \case
    Left ErrBuilderCudaNvrtcError{} -> True
    _ -> False

copyKernelSignature :: KernelSignature
copyKernelSignature =
    KernelSignature
        { extentParamName = "N"
        , tensorParamNames = ["A", "B"]
        }

addKernelSignature :: KernelSignature
addKernelSignature =
    KernelSignature
        { extentParamName = "N"
        , tensorParamNames = ["A", "B", "C"]
        }

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

addKernel :: String
addKernel =
    unlines
        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B, float* C) {"
        , "    int i = blockIdx.x * blockDim.x + threadIdx.x;"
        , "    if (i < N) {"
        , "        A[i] = B[i] + C[i];"
        , "    }"
        , "}"
        ]

malformedKernel :: String
malformedKernel =
    "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B) { this is not valid CUDA; }"
