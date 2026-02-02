module ISL.Internal.ScheduleConstraints.Ops (
    -- * Schedule Constraints Operations
    scheduleConstraintsOnDomain,
    scheduleConstraintsSetValidity,
    scheduleConstraintsSetProximity,
    scheduleConstraintsSetCoincidence,
    scheduleConstraintsSetContext,
    scheduleConstraintsComputeSchedule,
) where

import           Foreign.ForeignPtr                     (withForeignPtr)
import           ISL.Core                               (ISL, manage)
import           ISL.Internal.FFI
import           ISL.Internal.Map.Types                 (UnionMap (..))
import           ISL.Internal.Schedule.Types            (Schedule (..))
import           ISL.Internal.ScheduleConstraints.Types (ScheduleConstraints (..))
import           ISL.Internal.Set.Types                 (Set (..),
                                                         UnionSet (..))

-- | Create schedule constraints from a domain
scheduleConstraintsOnDomain :: UnionSet s -> ISL s (ScheduleConstraints s)
scheduleConstraintsOnDomain (UnionSet fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_uset_copy ptr
            c_sched_constraints_on_domain cptr
    manage c_sched_constraints_free "isl_schedule_constraints_on_domain" mk ScheduleConstraints

-- | Set validity constraints (dependencies that must be respected)
scheduleConstraintsSetValidity ::
    ScheduleConstraints s -> UnionMap s -> ISL s (ScheduleConstraints s)
scheduleConstraintsSetValidity (ScheduleConstraints scFP) (UnionMap umFP) = do
    let mk = withForeignPtr scFP $ \scPtr ->
            withForeignPtr umFP $ \umPtr -> do
                umCopy <- c_umap_copy umPtr
                c_sched_constraints_set_validity scPtr umCopy
    manage c_sched_constraints_free "isl_schedule_constraints_set_validity" mk ScheduleConstraints

-- | Set proximity constraints (hints for data locality)
scheduleConstraintsSetProximity ::
    ScheduleConstraints s -> UnionMap s -> ISL s (ScheduleConstraints s)
scheduleConstraintsSetProximity (ScheduleConstraints scFP) (UnionMap umFP) = do
    let mk = withForeignPtr scFP $ \scPtr ->
            withForeignPtr umFP $ \umPtr -> do
                umCopy <- c_umap_copy umPtr
                c_sched_constraints_set_proximity scPtr umCopy
    manage c_sched_constraints_free "isl_schedule_constraints_set_proximity" mk ScheduleConstraints

-- | Set coincidence constraints (hints for parallelization)
scheduleConstraintsSetCoincidence ::
    ScheduleConstraints s -> UnionMap s -> ISL s (ScheduleConstraints s)
scheduleConstraintsSetCoincidence (ScheduleConstraints scFP) (UnionMap umFP) = do
    let mk = withForeignPtr scFP $ \scPtr ->
            withForeignPtr umFP $ \umPtr -> do
                umCopy <- c_umap_copy umPtr
                c_sched_constraints_set_coincidence scPtr umCopy
    manage c_sched_constraints_free "isl_schedule_constraints_set_coincidence" mk ScheduleConstraints

-- | Set context (parameter constraints) for scheduling
scheduleConstraintsSetContext ::
    ScheduleConstraints s -> Set s -> ISL s (ScheduleConstraints s)
scheduleConstraintsSetContext (ScheduleConstraints scFP) (Set setFP) = do
    let mk = withForeignPtr scFP $ \scPtr ->
            withForeignPtr setFP $ \setPtr -> do
                setCopy <- c_set_copy setPtr
                c_sched_constraints_set_context scPtr setCopy
    manage c_sched_constraints_free "isl_schedule_constraints_set_context" mk ScheduleConstraints

-- | Compute an optimal schedule from constraints
scheduleConstraintsComputeSchedule :: ScheduleConstraints s -> ISL s (Schedule s)
scheduleConstraintsComputeSchedule (ScheduleConstraints scFP) = do
    let mk = withForeignPtr scFP $ \scPtr -> do
            c_sched_constraints_compute_schedule scPtr
    manage c_sched_free "isl_schedule_constraints_compute_schedule" mk Schedule
