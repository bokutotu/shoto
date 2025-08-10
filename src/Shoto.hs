module Shoto (compile) where

import           FrontendIR (FrontendIR (..), codegen, convert)

compile :: FrontendIR -> [String]
compile fir =
    let ir = convert fir
     in codegen ir
