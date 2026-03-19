module Polyhedral.Internal.Flow (
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

import           Polyhedral.Internal.Flow.Ops
import           Polyhedral.Internal.Flow.Types
