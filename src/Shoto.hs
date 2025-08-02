module Shoto (compile, toCuda, nvcc) where

import           FrontendIR     (FrontendIR)
import           System.Process (callProcess)

--
compile :: FrontendIR -> [String]
compile ir =
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

toCuda :: [String] -> String -> IO ()
toCuda code path = writeFile path (unlines code)

nvcc :: String -> String -> IO ()
nvcc soPath cudaPath = callProcess "nvcc" ["-Xcompiler", "-fPIC", "-shared", "-o", soPath, cudaPath]
