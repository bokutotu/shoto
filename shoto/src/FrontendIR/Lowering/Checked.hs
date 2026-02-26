module FrontendIR.Lowering.Checked (
    CheckedProgram (..),
    CheckedStmt (..),
) where

import qualified Data.Map.Strict  as Map
import           FrontendIR.Types (IterName, ParamName, TensorName)

data CheckedProgram = CheckedProgram
    { checkedParams :: [ParamName]
    , checkedIters :: [IterName]
    , checkedIterExtents :: Map.Map IterName ParamName
    , checkedStmts :: [CheckedStmt]
    }

data CheckedStmt
    = CAssign
        { cOutputTensor :: TensorName
        , cOutputIndex :: [IterName]
        , cLoads :: [(TensorName, [IterName])]
        }
    | CReduction
        { cOutputTensor :: TensorName
        , cOutputIndex :: [IterName]
        , cLoads :: [(TensorName, [IterName])]
        , cReductionAxes :: [IterName]
        }
