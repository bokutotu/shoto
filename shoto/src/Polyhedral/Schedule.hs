{-# LANGUAGE OverloadedRecordDot #-}

module Polyhedral.Schedule (
    computeSchedule,
) where

import           ISL              (ISL, Schedule, Set,
                                   scheduleConstraintsComputeSchedule,
                                   scheduleConstraintsOnDomain,
                                   scheduleConstraintsSetCoincidence,
                                   scheduleConstraintsSetContext,
                                   scheduleConstraintsSetProximity,
                                   scheduleConstraintsSetValidity)
import           Polyhedral.Types (Dependencies (..), Domain,
                                   IntoUnionMap (intoUnionMap),
                                   IntoUnionSet (intoUnionSet))

-- | 依存関係制約からスケジュールを計算
computeSchedule :: Set s -> Domain s -> Dependencies s -> ISL s (Schedule s)
computeSchedule ctx domain deps = do
    sc <- scheduleConstraintsOnDomain (intoUnionSet domain)
    sc' <- scheduleConstraintsSetContext sc ctx
    let must = intoUnionMap deps.legality
    sc'' <- scheduleConstraintsSetValidity sc' must
    sc''' <- scheduleConstraintsSetCoincidence sc'' must
    sc'''' <- scheduleConstraintsSetProximity sc''' (intoUnionMap deps.proximity)
    scheduleConstraintsComputeSchedule sc''''
