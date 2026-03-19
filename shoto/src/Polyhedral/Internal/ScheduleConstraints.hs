module Polyhedral.Internal.ScheduleConstraints (
    -- * Types
    ScheduleConstraints (..),

    -- * Schedule Constraints Operations
    scheduleConstraintsOnDomain,
    scheduleConstraintsSetValidity,
    scheduleConstraintsSetProximity,
    scheduleConstraintsSetCoincidence,
    scheduleConstraintsSetContext,
    scheduleConstraintsComputeSchedule,
) where

import           Polyhedral.Internal.ScheduleConstraints.Ops
import           Polyhedral.Internal.ScheduleConstraints.Types
