module ISL.ScheduleConstraints (
    -- * Types
    ScheduleConstraints (..),

    -- * Schedule Constraints Operations
    scheduleConstraintsOnDomain,
    scheduleConstraintsSetValidity,
    scheduleConstraintsSetProximity,
    scheduleConstraintsSetCoincidence,
    scheduleConstraintsComputeSchedule,
) where

import           ISL.Internal.ScheduleConstraints.Ops
import           ISL.Internal.ScheduleConstraints.Types
