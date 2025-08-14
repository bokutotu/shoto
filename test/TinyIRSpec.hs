module TinyIRSpec where

import qualified Data.ByteString.Char8 as BS
import           Foreign               (ForeignPtr, withForeignPtr)
import           Foreign.C             (CFloat)
import           Foreign.Marshal       (alloca)
import           Foreign.Ptr           (Ptr, castPtr)
import           Foreign.Storable      (poke)
import           Runtime
import           Test.Hspec
import           TinyIR

spec :: Spec
spec = describe "TinyIR ElementWise Codegen" $ do
    describe "Binary operations" $ do
        it "generates correct Add kernel" $
            testBinaryOp Add [1.0, 2.0, 3.0] [4.0, 5.0, 6.0] [5.0, 7.0, 9.0]
        it "generates correct Sub kernel" $
            testBinaryOp Sub [5.0, 7.0, 9.0] [2.0, 3.0, 4.0] [3.0, 4.0, 5.0]
        it "generates correct Mul kernel" $
            testBinaryOp Mul [2.0, 3.0, 4.0] [5.0, 6.0, 7.0] [10.0, 18.0, 28.0]
        it "generates correct Div kernel" $
            testBinaryOp Div [10.0, 15.0, 20.0] [2.0, 3.0, 4.0] [5.0, 5.0, 5.0]

testBinaryOp :: BinaryTy -> [CFloat] -> [CFloat] -> [CFloat] -> IO ()
testBinaryOp ty a b expected = do
    let size = length a
        code = BS.pack . unlines $ codegenElementWise (Binary ty) []
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
                        { gridDimX = (fromIntegral size + 255) `div` 256
                        , gridDimY = 1
                        , gridDimZ = 1
                        , blockDimX = 256
                        , blockDimY = 1
                        , blockDimZ = 1
                        , sharedMemBytes = 0
                        }
            withKernelArgs3 (ptr aGpu) (ptr bGpu) (ptr cGpu) (fromIntegral size) $ \args ->
                launchKernel func cfg args

    cCpu <- copyToCpu cGpu
    cCpu `shouldBe` expected

withKernelArgs3 ::
    ForeignPtr a ->
    ForeignPtr b ->
    ForeignPtr c ->
    Int ->
    ([Ptr ()] -> IO r) ->
    IO r
withKernelArgs3 aF bF cF n action =
    withForeignPtr aF $ \aPtr ->
        withForeignPtr bF $ \bPtr ->
            withForeignPtr cF $ \cPtr ->
                alloca $ \aPtrPtr -> alloca $ \bPtrPtr -> alloca $ \cPtrPtr -> alloca $ \nPtr -> do
                    poke aPtrPtr aPtr
                    poke bPtrPtr bPtr
                    poke cPtrPtr cPtr
                    poke nPtr n
                    action [castPtr aPtrPtr, castPtr bPtrPtr, castPtr cPtrPtr, castPtr nPtr]