module FrontendIR.Lowering (
    CheckedProgram,
    checkProgram,
    lowerToRaw,
) where

import           FrontendIR.Lowering.Checked    (CheckedProgram)
import           FrontendIR.Lowering.Checker    (checkProgram)
import           FrontendIR.Lowering.RawBuilder (lowerToRaw)
