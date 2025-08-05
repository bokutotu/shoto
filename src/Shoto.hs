module Shoto (compile) where

import           FrontendIR (FrontendIR (..), opStr)

--
compile :: FrontendIR -> [String]
compile ir =
    let op = opStr ir
     in [ "extern \"C\" __global__ void kernel(float *__restrict__ a, float* __restrict__ b, float* __restrict__ c) {"
        , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        , "  c[idx] = a[idx] " ++ op ++ " b[idx];"
        , "}"
        ]
