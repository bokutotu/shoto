{-# LANGUAGE ForeignFunctionInterface #-}

module ArithmeticSpec where

import           Data.List      (isInfixOf)
import           Foreign        (FunPtr, Ptr, withForeignPtr)
import           Foreign.C      (CFloat)
import           FrontendIR     (FrontendIR (..))
import           Runtime        (GpuPtr (..), copyToCpu, copyToGpu, withKernel)
import           Shoto
import           System.Exit    (ExitCode (..))
import           System.Process (readProcessWithExitCode)
import           Tensor         (Shape (..), Tensor (..))
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
testBinaryOp op opName a b expected = do
    let size = length a
        aT = Tensor (Shape [size]) (Shape [1])
        bT = Tensor (Shape [size]) (Shape [1])
        ir = op aT bT
        code = compile ir
        fileName = "/tmp/" ++ opName ++ ".cu"
        libName = "/tmp/" ++ opName ++ ".so"

    -- Compile CUDA code
    toCuda code fileName
    nvcc libName fileName

    -- Check function exists
    hasFunc <- hasFunction libName "add" -- Note: function name is still "add" in current implementation
    hasFunc `shouldBe` True

    -- Prepare GPU memory
    let c = replicate size 0.0 :: [CFloat]
    aGpu <- copyToGpu a
    bGpu <- copyToGpu b
    cGpu <- copyToGpu c

    -- Execute kernel
    withKernel libName "kernel" $ \fptr -> do
        let kernel = mkKernel fptr
        withForeignPtr (ptr aGpu) $ \aPtr ->
            withForeignPtr (ptr bGpu) $ \bPtr ->
                withForeignPtr (ptr cGpu) $ \cPtr ->
                    kernel aPtr bPtr cPtr

    -- Verify results
    cCpu <- copyToCpu cGpu
    cCpu `shouldBe` expected
