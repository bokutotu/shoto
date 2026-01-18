module ISL.Flow (
    -- * Types
    UnionAccessInfo (..),
    UnionFlow (..),

    -- * UnionAccessInfo Operations
    unionAccessInfoFromSink,
    unionAccessInfoSetMustSource,
    unionAccessInfoSetMaySource,
    unionAccessInfoSetScheduleMap,
    unionAccessInfoComputeFlow,

    -- * UnionFlow Operations
    unionFlowGetMustDependence,
    unionFlowGetMayDependence,
) where

import           ISL.Internal.Flow.Ops
import           ISL.Internal.Flow.Types
