module FrontendIR.Lowering (
    lowerProgram,
) where

import           FrontendIR.Lowering.Parse      (parseProgram)
import           FrontendIR.Lowering.RawBuilder (lowerToRaw)
import           FrontendIR.Types               (FrontendError, Program)
import           Polyhedral.Parse               (RawPolyhedralModel)

lowerProgram :: Program -> Either FrontendError RawPolyhedralModel
lowerProgram program = lowerToRaw <$> parseProgram program
