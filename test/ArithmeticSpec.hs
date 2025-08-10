module ArithmeticSpec where

import qualified Data.ByteString.Char8 as BS
import           Foreign               (ForeignPtr, withForeignPtr)
import           Foreign.C             (CFloat)
import           Foreign.Marshal       (alloca)
import           Foreign.Ptr           (Ptr, castPtr)
import           Foreign.Storable      (poke)
import           FrontendIR            (EOpTy (..), FTensor (..),
                                        FrontendIR (..), Op (..), codegen)
import           Runtime
import           Test.Hspec

spec :: Spec
spec = describe "FrontendIR Arithmetic Kernel" $ do
    describe "Addition" $ do
        it "computes element-wise addition" $
            testBinaryOp Add [1.0, 2.0] [2.0, 4.0] [3.0, 6.0]
        it "handles larger arrays" $
            testBinaryOp Add [1.0, 2.0, 3.0, 4.0] [5.0, 6.0, 7.0, 8.0] [6.0, 8.0, 10.0, 12.0]

    describe "Subtraction" $ do
        it "computes element-wise subtraction" $
            testBinaryOp Sub [5.0, 8.0] [2.0, 3.0] [3.0, 5.0]
        it "handles negative results" $
            testBinaryOp Sub [1.0, 2.0] [3.0, 4.0] [-2.0, -2.0]

    describe "Multiplication" $ do
        it "computes element-wise multiplication" $
            testBinaryOp Mul [2.0, 3.0] [4.0, 5.0] [8.0, 15.0]
        it "handles multiplication by zero" $
            testBinaryOp Mul [5.0, 0.0] [0.0, 7.0] [0.0, 0.0]

    describe "Division" $ do
        it "computes element-wise division" $
            testBinaryOp Div [10.0, 15.0] [2.0, 3.0] [5.0, 5.0]
        it "handles fractional results" $
            testBinaryOp Div [7.0, 5.0] [2.0, 2.0] [3.5, 2.5]

testBinaryOp :: EOpTy -> [CFloat] -> [CFloat] -> [CFloat] -> IO ()
testBinaryOp ty a b expected = do
    let size = length a
        -- NOTE: outputs は codegen の空白欠落を回避するため " c" としています
        ir =
            FrontendIR
                { tensors = [FTensor "a" [size], FTensor "b" [size], FTensor "c" [size]]
                , inputs = ["a", "b"]
                , outputs = [" c"]
                , ops = [ElementWise{name = "op", a = "a", b = "b", c = "c", ty = ty}]
                }
        code = BS.pack . unlines $ codegen ir
        cInit = replicate size 0.0 :: [CFloat]
        compileOptions =
            [ "--gpu-architecture=compute_80"
            , "-default-device"
            , "--use_fast_math"
            , "--fmad=true"
            ]

    -- Prepare device buffers
    aGpu <- copyToGpu a
    bGpu <- copyToGpu b
    cGpu <- copyToGpu cInit

    withCudaKernel
        0
        code
        (BS.pack "kernel.cu")
        (map BS.pack compileOptions)
        (BS.pack "kernel")
        $ \func -> do
            let cfg =
                    KernelLaunchConfig
                        { gridDimX = fromIntegral size
                        , gridDimY = 1
                        , gridDimZ = 1
                        , blockDimX = 1
                        , blockDimY = 1
                        , blockDimZ = 1
                        , sharedMemBytes = 0
                        }
            withKernelArgs3 (ptr aGpu) (ptr bGpu) (ptr cGpu) $ \args ->
                launchKernel func cfg args

    cCpu <- copyToCpu cGpu
    cCpu `shouldBe` expected

withKernelArgs3 ::
    ForeignPtr a ->
    ForeignPtr b ->
    ForeignPtr c ->
    ([Ptr ()] -> IO r) ->
    IO r
withKernelArgs3 aF bF cF action =
    withForeignPtr aF $ \aPtr ->
        withForeignPtr bF $ \bPtr ->
            withForeignPtr cF $ \cPtr ->
                alloca $ \aPtrPtr -> alloca $ \bPtrPtr -> alloca $ \cPtrPtr -> do
                    poke aPtrPtr aPtr
                    poke bPtrPtr bPtr
                    poke cPtrPtr cPtr
                    action [castPtr aPtrPtr, castPtr bPtrPtr, castPtr cPtrPtr]
