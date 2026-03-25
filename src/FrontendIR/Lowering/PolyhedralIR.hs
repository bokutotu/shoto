module FrontendIR.Lowering.PolyhedralIR (
    StmtName (..),
    stmtNameToString,
    ScheduleAxis (..),
    PolyDomain (..),
    PolySchedule (..),
    PolyAccess (..),
    PolyStmt (..),
    PolyProgram (..),
) where

import qualified Data.Map.Strict  as Map
import           FrontendIR.Types (IterName, ParamName, TensorName)
import           IR.Name          (StmtName (..), stmtNameToString)

data ScheduleAxis
    = PadAxis Int
    | IterAxis IterName
    deriving (Eq, Ord, Show)

data PolyDomain = PolyDomain
    { dStmtName :: StmtName
    , dIters :: [IterName]
    }
    deriving (Eq, Ord, Show)

data PolySchedule = PolySchedule
    { sStmtName :: StmtName
    , sIters :: [IterName]
    , sAxes :: [ScheduleAxis]
    }
    deriving (Eq, Ord, Show)

data PolyAccess = PolyAccess
    { aStmtName :: StmtName
    , aIters :: [IterName]
    , aTensor :: TensorName
    , aIndices :: [IterName]
    }
    deriving (Eq, Ord, Show)

data PolyStmt = PolyStmt
    { stmtDomain :: PolyDomain
    , stmtSchedule :: PolySchedule
    , stmtReads :: [PolyAccess]
    , stmtWrite :: PolyAccess
    , stmtIsReduction :: Bool
    }
    deriving (Eq, Show)

data PolyProgram = PolyProgram
    { pParams :: [ParamName]
    , pIterExtents :: Map.Map IterName ParamName
    , pStmts :: [PolyStmt]
    }
    deriving (Eq, Show)
