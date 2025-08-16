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
spec = describe "TinyIR Codegen" $ do
    describe "Binary operations" $ do
        it "generates correct Add kernel" $
            testBinaryOp Add [1.0, 2.0, 3.0] [4.0, 5.0, 6.0] [5.0, 7.0, 9.0]
        it "generates correct Sub kernel" $
            testBinaryOp Sub [5.0, 7.0, 9.0] [2.0, 3.0, 4.0] [3.0, 4.0, 5.0]
        it "generates correct Mul kernel" $
            testBinaryOp Mul [2.0, 3.0, 4.0] [5.0, 6.0, 7.0] [10.0, 18.0, 28.0]
        it "generates correct Div kernel" $
            testBinaryOp Div [10.0, 15.0, 20.0] [2.0, 3.0, 4.0] [5.0, 5.0, 5.0]
    
    describe "Reduce operations" $ do
        it "reduces all elements to scalar (Sum)" $
            testReduceAll Sum [1.0, 2.0, 3.0, 4.0, 5.0] 15.0
        it "reduces all elements to scalar (Max)" $
            testReduceAll Max [1.0, 5.0, 3.0, 2.0, 4.0] 5.0
        it "reduces all elements to scalar (Min)" $
            testReduceAll Min [5.0, 2.0, 8.0, 1.0, 3.0] 1.0
        
        it "reduces along axis (Sum)" $
            -- Shape [2, 3] -> axis=1 -> [2]
            -- [[1,2,3], [4,5,6]] -> [6, 15]
            testReduceAxis Sum 1 [Static 2, Static 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [6.0, 15.0]

testBinaryOp :: BinaryTy -> [CFloat] -> [CFloat] -> [CFloat] -> IO ()
testBinaryOp ty a b expected = do
    let size = length a
        code = codegenElementWise (Binary ty) []
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

testReduceAll :: ReduceOp -> [CFloat] -> CFloat -> IO ()
testReduceAll op input expected = do
    let size = length input
        code = codegenReduce op Nothing []
        compileOptions =
            [ "--gpu-architecture=compute_80"
            , "-default-device"
            , "--use_fast_math"
            , "--fmad=true"
            ]
    
    -- Prepare device buffers
    inputGpu <- copyToGpu input
    outputGpu <- copyToGpu [0.0 :: CFloat]  -- Single element for scalar output
    
    withCudaKernel
        0
        code
        (BS.pack "kernel.cu")
        (map BS.pack compileOptions)
        (BS.pack "kernel")
        $ \func -> do
            let cfg =
                    KernelLaunchConfig
                        { gridDimX = 1
                        , gridDimY = 1
                        , gridDimZ = 1
                        , blockDimX = 1
                        , blockDimY = 1
                        , blockDimZ = 1
                        , sharedMemBytes = 0
                        }
            withKernelArgs2 (ptr inputGpu) (ptr outputGpu) (fromIntegral size) $ \args ->
                launchKernel func cfg args
    
    [result] <- copyToCpu outputGpu
    result `shouldBe` expected

testReduceAxis :: ReduceOp -> Int -> Shape -> [CFloat] -> [CFloat] -> IO ()
testReduceAxis op axis shape input expected = do
    let totalSize = product [n | Static n <- shape]
        outputSize = product [n | (i, Static n) <- zip [0..] shape, i /= axis]
        reduceSize = case shape !! axis of Static n -> n; _ -> error "Dynamic shape not supported"
        innerSize = product [n | (i, Static n) <- zip [0..] shape, i > axis]
        outerStride = reduceSize * innerSize
        
        code = codegenReduce op (Just axis) shape
        compileOptions =
            [ "--gpu-architecture=compute_80"
            , "-default-device"
            , "--use_fast_math"
            , "--fmad=true"
            ]
    
    -- Prepare device buffers
    inputGpu <- copyToGpu input
    outputGpu <- copyToGpu (replicate outputSize 0.0)
    
    withCudaKernel
        0
        code
        (BS.pack "kernel.cu")
        (map BS.pack compileOptions)
        (BS.pack "kernel")
        $ \func -> do
            let cfg =
                    KernelLaunchConfig
                        { gridDimX = (fromIntegral outputSize + 255) `div` 256
                        , gridDimY = 1
                        , gridDimZ = 1
                        , blockDimX = 256
                        , blockDimY = 1
                        , blockDimZ = 1
                        , sharedMemBytes = 0
                        }
            withKernelArgs6 
                (ptr inputGpu) 
                (ptr outputGpu) 
                (fromIntegral outerStride)
                (fromIntegral reduceSize)
                (fromIntegral innerSize)
                (fromIntegral outputSize) $ \args ->
                    launchKernel func cfg args
    
    result <- copyToCpu outputGpu
    result `shouldBe` expected

withKernelArgs2 ::
    ForeignPtr a ->
    ForeignPtr b ->
    Int ->
    ([Ptr ()] -> IO r) ->
    IO r
withKernelArgs2 aF bF n action =
    withForeignPtr aF $ \aPtr ->
        withForeignPtr bF $ \bPtr ->
            alloca $ \aPtrPtr -> alloca $ \bPtrPtr -> alloca $ \nPtr -> do
                poke aPtrPtr aPtr
                poke bPtrPtr bPtr
                poke nPtr n
                action [castPtr aPtrPtr, castPtr bPtrPtr, castPtr nPtr]

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

withKernelArgs6 ::
    ForeignPtr a ->
    ForeignPtr b ->
    Int -> Int -> Int -> Int ->
    ([Ptr ()] -> IO r) ->
    IO r
withKernelArgs6 aF bF n1 n2 n3 n4 action =
    withForeignPtr aF $ \aPtr ->
        withForeignPtr bF $ \bPtr ->
            alloca $ \aPtrPtr -> alloca $ \bPtrPtr -> 
            alloca $ \n1Ptr -> alloca $ \n2Ptr -> alloca $ \n3Ptr -> alloca $ \n4Ptr -> do
                poke aPtrPtr aPtr
                poke bPtrPtr bPtr
                poke n1Ptr n1
                poke n2Ptr n2
                poke n3Ptr n3
                poke n4Ptr n4
                action [castPtr aPtrPtr, castPtr bPtrPtr, castPtr n1Ptr, castPtr n2Ptr, castPtr n3Ptr, castPtr n4Ptr]