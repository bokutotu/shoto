module AddSpec where

import           Data.List      (isInfixOf)
import           Shoto
import           System.Exit    (ExitCode (..))
import           System.Process (readProcessWithExitCode)
import           Test.Hspec

checkSymbols :: FilePath -> IO [String]
checkSymbols soPath = do
    (exitCode, stdout, stderr) <- readProcessWithExitCode "nm" ["-D", soPath] ""
    case exitCode of
        ExitSuccess -> return $ lines stdout
        _ -> error $ "nm failed" ++ stderr

hasFunction :: FilePath -> String -> IO Bool
hasFunction soPath funcName = any (isInfixOf funcName) <$> checkSymbols soPath

spec :: Spec
spec = describe "Shoto Compiler Add Test" $ do
    it "generate Add Kenel" $ do
        compile `shouldBe` expectedCode
    it "codegen and comple" $ do
        let code = compile
            fileName = "/tmp/add.cu"
            libName = "/tmp/add.so"
        toCuda code fileName
        nvcc libName fileName
        hasAdd <- hasFunction libName "add"
        hasAdd `shouldBe` True
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
