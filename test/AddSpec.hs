{-# LANGUAGE ForeignFunctionInterface #-}

module AddSpec where

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

-- ファイルの上部に追加
foreign import ccall "dynamic"
    mkAdd ::
        FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> IO ()) ->
        (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> IO ())

spec :: Spec
spec = describe "Shoto Compiler Add Test" $ do
    it "generate Add Kenel" $ do
        let a = Tensor (Shape [1]) (Shape [1])
            b = Tensor (Shape [1]) (Shape [1])
            ir = Add a b
        compile ir `shouldBe` expectedCode
    it "codegen and comple" $ do
        let aT = Tensor (Shape [1]) (Shape [1])
            bT = Tensor (Shape [1]) (Shape [1])
            ir = Add aT bT
            code = compile ir
            fileName = "/tmp/add.cu"
            libName = "/tmp/add.so"
        toCuda code fileName
        nvcc libName fileName
        hasAdd <- hasFunction libName "add"
        hasAdd `shouldBe` True
        let a = [1.0] :: [CFloat]
            b = [2.0] :: [CFloat]
            c = [0.0] :: [CFloat]
        aGpu <- copyToGpu a
        bGpu <- copyToGpu b
        cGpu <- copyToGpu c
        withKernel "/tmp/add.so" "add" $ \fptr -> do
            let add = mkAdd fptr
            withForeignPtr (ptr aGpu) $ \aPtr ->
                withForeignPtr (ptr bGpu) $ \bPtr ->
                    withForeignPtr (ptr cGpu) $ \cPtr ->
                        add aPtr bPtr cPtr
        cCpu <- copyToCpu cGpu
        cCpu `shouldBe` [3]
  where
    expectedCode =
        [ "#include <cuda_runtime.h>"
        , "__global__ void add_kernel(float *__restrict__ a, float* __restrict__ b, float* __restrict__ c) {"
        , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        , "  c[idx] = a[idx] + b[idx];"
        , "}"
        , "extern \"C\" void add(float *a, float *b, float *c) {"
        , "  dim3 grid(1);"
        , "  dim3 block(1);"
        , "  add_kernel<<< grid, block >>> (a, b, c);"
        , "  cudaDeviceSynchronize();"
        , "}"
        ]
