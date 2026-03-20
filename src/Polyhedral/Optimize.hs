{-# LANGUAGE LambdaCase #-}

module Polyhedral.Optimize (
    ScheduleOptimization (..),
    applyScheduleOptimizations,
) where

import           Control.Monad        (foldM, unless, when)
import           Control.Monad.Except (catchError, throwError)
import           Data.List            (transpose)
import           Polyhedral.Error     (OptimizeError (..), PolyhedralError (..))
import           Polyhedral.Internal  (ISL, Schedule, ScheduleNode,
                                       ScheduleNodeType (..),
                                       scheduleMapScheduleNodeBottomUp,
                                       scheduleNodeBandNMember,
                                       scheduleNodeBandPermuteMembers,
                                       scheduleNodeBandTile, scheduleNodeChild,
                                       scheduleNodeGetType,
                                       scheduleNodeNChildren,
                                       scheduleNodeParent)

data ScheduleOptimization
    = LoopInterchange [Int]
    | Tile [[Int]]
    deriving (Eq, Show)

applyScheduleOptimizations :: [ScheduleOptimization] -> Schedule s -> ISL s (Schedule s)
applyScheduleOptimizations opts sched =
    catchError
        ( foldM
            (flip applyScheduleOptimization)
            sched
            opts
        )
        ( \case
            InternalIslError islErr ->
                throwError (PolyhedralOptimizeError OptimizeInternalFailure (Just islErr))
            other -> throwError other
        )

applyScheduleOptimization :: ScheduleOptimization -> Schedule s -> ISL s (Schedule s)
applyScheduleOptimization (LoopInterchange permutation) sched =
    applyToBandNodes (scheduleNodeBandPermuteMembers permutation) sched
applyScheduleOptimization (Tile tileSizesByAxis) sched = do
    levelMajorSizes <- tileSizesToLevelMajor tileSizesByAxis
    applyToBandNodes (tileBand levelMajorSizes) sched
  where
    axisCount = length tileSizesByAxis

    tileBand levels node = do
        nMember <- scheduleNodeBandNMember node
        unless (nMember == axisCount) $
            throwError (PolyhedralOptimizeError (OptimizeBandRankMismatch axisCount nMember) Nothing)
        tileBandMultiLevel levels node

applyToBandNodes ::
    (ScheduleNode s -> ISL s (ScheduleNode s)) ->
    Schedule s ->
    ISL s (Schedule s)
applyToBandNodes transform sched =
    scheduleMapScheduleNodeBottomUp sched $ \node -> do
        nodeType <- scheduleNodeGetType node
        case nodeType of
            ScheduleNodeBand -> transform node
            _ -> pure node

tileBandMultiLevel :: [[Int]] -> ScheduleNode s -> ISL s (ScheduleNode s)
tileBandMultiLevel [] node = pure node
tileBandMultiLevel (sizes : rest) node = do
    outerBand <- scheduleNodeBandTile sizes node
    case rest of
        [] -> pure outerBand
        _ -> do
            nChildren <- scheduleNodeNChildren outerBand
            when (nChildren /= 1) $
                throwError (PolyhedralOptimizeError (OptimizeTiledBandExpectedOneChild nChildren) Nothing)
            inner <- scheduleNodeChild outerBand 0
            innerType <- scheduleNodeGetType inner
            unless (innerType == ScheduleNodeBand) $
                throwError (PolyhedralOptimizeError OptimizeTiledBandChildNotBand Nothing)
            inner' <- tileBandMultiLevel rest inner
            scheduleNodeParent inner'

tileSizesToLevelMajor :: [[Int]] -> ISL s [[Int]]
tileSizesToLevelMajor [] =
    throwError (PolyhedralOptimizeError OptimizeTileNoAxis Nothing)
tileSizesToLevelMajor ([] : _) =
    throwError (PolyhedralOptimizeError OptimizeTileEmptyLevel Nothing)
tileSizesToLevelMajor sizesByAxis@(firstAxis : restAxes) = do
    when (any null restAxes) $
        throwError (PolyhedralOptimizeError OptimizeTileEmptyLevel Nothing)
    let levelCount = length firstAxis
    unless (all (\sizes -> length sizes == levelCount) sizesByAxis) $
        throwError (PolyhedralOptimizeError OptimizeTileLevelCountMismatch Nothing)
    unless (all (all (> 0)) sizesByAxis) $
        throwError (PolyhedralOptimizeError OptimizeTileNonPositiveSize Nothing)
    pure (transpose sizesByAxis)
