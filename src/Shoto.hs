module Shoto (compile) where

import           FrontendIR (FrontendIR (..), codegen)

compile :: FrontendIR -> [String]
compile = codegen
