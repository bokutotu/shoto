{-# LANGUAGE LambdaCase #-}

module Shoto (compile, toCuda, nvcc) where

import           FrontendIR     (FrontendIR (..))
import           System.Process (callProcess)
import           Tensor         (Tensor (..), shapeIdx)

--
compile :: FrontendIR -> [String]
compile = \case
    Add Tensor{shape = aShape} Tensor{shape = bShape} ->
        [ "#include <cuda_runtime.h>"
        , "__global__ void add_kernel(float *__restrict__ a, float* __restrict__ b, float* __restrict__ c) {"
        , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        , "  c[idx] = a[idx] + b[idx];"
        , "}"
        , "extern \"C\" void add(float *a, float *b, float *c) {"
        , "  dim3 grid(" ++ show (shapeIdx aShape 0) ++ ");"
        , "  dim3 block(1);"
        , "  add_kernel<<< grid, block >>> (a, b, c);"
        , "  cudaDeviceSynchronize();"
        , "}"
        ]
    _ -> []

toCuda :: [String] -> String -> IO ()
toCuda code path = writeFile path (unlines code)

nvcc :: String -> String -> IO ()
nvcc soPath cudaPath = callProcess "nvcc" ["-Xcompiler", "-fPIC", "-shared", "-o", soPath, cudaPath]
