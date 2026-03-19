module FrontendIR (
    module FrontendIR.Types,
    lowerProgram,
) where

import           FrontendIR.Lowering (lowerProgram)
import           FrontendIR.Types
