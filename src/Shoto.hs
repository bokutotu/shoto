module Shoto (compile, toCuda, nvcc) where

import           FrontendIR     (FrontendIR (..), gridDim, opStr)
import           System.Process (callProcess)

--
compile :: FrontendIR -> [String]
compile ir =
    let op = opStr ir
        grid = gridDim ir
     in [ "#include <cuda_runtime.h>"
        , "__global__ void gpu_kernel(float *__restrict__ a, float* __restrict__ b, float* __restrict__ c) {"
        , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        , "  c[idx] = a[idx] " ++ op ++ " b[idx];"
        , "}"
        , "extern \"C\" void kernel(float *a, float *b, float *c) {"
        , "  dim3 grid(" ++ show grid ++ ");"
        , "  dim3 block(1);"
        , "  gpu_kernel<<< grid, block >>> (a, b, c);"
        , "  cudaDeviceSynchronize();"
        , "}"
        ]

toCuda :: [String] -> String -> IO ()
toCuda code path = writeFile path (unlines code)

nvcc :: String -> String -> IO ()
nvcc soPath cudaPath = callProcess "nvcc" ["-Xcompiler", "-fPIC", "-shared", "-o", soPath, cudaPath]
