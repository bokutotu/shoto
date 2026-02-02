{-# LANGUAGE OverloadedRecordDot #-}

module Polyhedral.Schedule (
    computeSchedule,
) where

import           ISL              (ISL, Schedule,
                                   scheduleConstraintsComputeSchedule,
                                   scheduleConstraintsOnDomain,
                                   scheduleConstraintsSetCoincidence,
                                   scheduleConstraintsSetProximity,
                                   scheduleConstraintsSetValidity)
import           Polyhedral.Types (Dependencies (..), Domain,
                                   IntoUnionMap (intoUnionMap),
                                   IntoUnionSet (intoUnionSet))

-- | 依存関係制約からスケジュールを計算
computeSchedule :: Domain s -> Dependencies s -> ISL s (Schedule s)
computeSchedule domain deps = do
    sc <- scheduleConstraintsOnDomain (intoUnionSet domain)
    sc' <- scheduleConstraintsSetValidity sc (intoUnionMap deps.validity)
    sc'' <- scheduleConstraintsSetCoincidence sc' (intoUnionMap deps.coincidence)
    sc''' <- scheduleConstraintsSetProximity sc'' (intoUnionMap deps.proximity)
    scheduleConstraintsComputeSchedule sc'''
