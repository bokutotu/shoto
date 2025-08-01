module Shoto (compile) where

--
compile :: [String]
compile =
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
