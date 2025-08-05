module ArithmeticSpec where

import qualified Data.ByteString.Char8 as BS
import           Data.List             (isInfixOf)
import           Foreign               (FunPtr, Ptr, withForeignPtr)
import           Foreign.C             (CFloat)
import           Foreign.Marshal       (alloca)
import           Foreign.Ptr           (castPtr)
import           Foreign.Storable      (poke)
import           FrontendIR            (FrontendIR (..))
import           Runtime
import           Shoto                 (compile)
import           System.Exit           (ExitCode (..))
import           System.Process        (readProcessWithExitCode)
import           Tensor                (Shape (..), Tensor (..))
import           Test.Hspec

checkSymbols :: FilePath -> IO [String]
checkSymbols soPath = do
    (exitCode, stdout, stderr) <- readProcessWithExitCode "nm" ["-D", soPath] ""
    case exitCode of
        ExitSuccess -> return $ lines stdout
        _ -> error $ "nm failed" ++ stderr

hasFunction :: FilePath -> String -> IO Bool
hasFunction soPath funcName = any (isInfixOf funcName) <$> checkSymbols soPath

foreign import ccall "dynamic"
    mkKernel ::
        FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> IO ()) ->
        (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> IO ())

spec :: Spec
spec = describe "Shoto Compiler Arithmetic Operations Test" $ do
    describe "Addition" $ do
        it "generates correct CUDA code and computes addition" $ do
            testBinaryOp Add "add" [1.0, 2.0] [2.0, 4.0] [3.0, 6.0]
        it "handles larger arrays" $ do
            testBinaryOp Add "add" [1.0, 2.0, 3.0, 4.0] [5.0, 6.0, 7.0, 8.0] [6.0, 8.0, 10.0, 12.0]

    describe "Subtraction" $ do
        it "generates correct CUDA code and computes subtraction" $ do
            testBinaryOp Sub "sub" [5.0, 8.0] [2.0, 3.0] [3.0, 5.0]
        it "handles negative results" $ do
            testBinaryOp Sub "sub" [1.0, 2.0] [3.0, 4.0] [-2.0, -2.0]

    describe "Multiplication" $ do
        it "generates correct CUDA code and computes multiplication" $ do
            testBinaryOp Mul "mul" [2.0, 3.0] [4.0, 5.0] [8.0, 15.0]
        it "handles multiplication by zero" $ do
            testBinaryOp Mul "mul" [5.0, 0.0] [0.0, 7.0] [0.0, 0.0]

    describe "Division" $ do
        it "generates correct CUDA code and computes division" $ do
            testBinaryOp Div "div" [10.0, 15.0] [2.0, 3.0] [5.0, 5.0]
        it "handles division resulting in fractions" $ do
            testBinaryOp Div "div" [7.0, 5.0] [2.0, 2.0] [3.5, 2.5]

testBinaryOp ::
    (Tensor -> Tensor -> FrontendIR) -> String -> [CFloat] -> [CFloat] -> [CFloat] -> IO ()
testBinaryOp op _ a b expected = do
    let size = length a
        aT = Tensor (Shape [size]) (Shape [size])
        bT = Tensor (Shape [size]) (Shape [size])
        ir = op aT bT
        code = BS.pack $ unlines $ compile ir
        c = replicate size 0.0 :: [CFloat]

        compileOptions =
            [ "--gpu-architecture=compute_70"
            , "-default-device" -- デフォルトデバイス設定
            , "--use_fast_math" -- 高速数学関数
            , "--fmad=true" -- FMA命令の使用
            ]
    
    -- Prepare GPU memory outside of withCudaKernel
    aGpu <- copyToGpu a
    bGpu <- copyToGpu b
    cGpu <- copyToGpu c
    
    withCudaKernel
        0
        code
        (BS.pack "kernel.cu")
        (map BS.pack compileOptions)
        (BS.pack "kernel")
        $ \func -> do
            let config =
                    KernelLaunchConfig
                        { gridDimX = fromIntegral size
                        , gridDimY = 1
                        , gridDimZ = 1
                        , blockDimX = 1
                        , blockDimY = 1
                        , blockDimZ = 1
                        , sharedMemBytes = 0
                        }
            -- Execute kernel
            withForeignPtr (ptr aGpu) $ \aPtr ->
                withForeignPtr (ptr bGpu) $ \bPtr ->
                    withForeignPtr (ptr cGpu) $ \cPtr -> do
                        -- Need to pass addresses of the pointers, not the pointers themselves
                        alloca $ \aPtrPtr -> alloca $ \bPtrPtr -> alloca $ \cPtrPtr -> do
                            poke aPtrPtr aPtr
                            poke bPtrPtr bPtr
                            poke cPtrPtr cPtr
                            launchKernel func config [castPtr aPtrPtr, castPtr bPtrPtr, castPtr cPtrPtr]
    
    -- Verify results
    cCpu <- copyToCpu cGpu
    cCpu `shouldBe` expected
