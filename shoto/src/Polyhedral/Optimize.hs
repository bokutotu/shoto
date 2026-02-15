module Polyhedral.Optimize (
    ScheduleOptimization (..),
    applyScheduleOptimizations,
) where

import           Control.Monad (foldM, unless, when)
import           Data.List     (transpose)
import           ISL           (ISL, Schedule, ScheduleNode,
                                ScheduleNodeType (..),
                                scheduleMapScheduleNodeBottomUp,
                                scheduleNodeBandNMember,
                                scheduleNodeBandPermuteMembers,
                                scheduleNodeBandTile, scheduleNodeChild,
                                scheduleNodeGetType, scheduleNodeNChildren,
                                scheduleNodeParent, throwISL)

data ScheduleOptimization
    = LoopInterchange [Int]
    | Tile [[Int]]
    deriving (Eq, Show)

applyScheduleOptimizations :: [ScheduleOptimization] -> Schedule s -> ISL s (Schedule s)
applyScheduleOptimizations opts sched =
    foldM
        (\acc opt -> applyScheduleOptimization opt acc)
        sched
        opts

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
            throwISL "applyScheduleOptimization(Tile): band rank mismatch"
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
                throwISL "tileBandMultiLevel: tiled band is expected to have one child"
            inner <- scheduleNodeChild outerBand 0
            innerType <- scheduleNodeGetType inner
            unless (innerType == ScheduleNodeBand) $
                throwISL "tileBandMultiLevel: tiled band child is expected to be a band"
            inner' <- tileBandMultiLevel rest inner
            scheduleNodeParent inner'

tileSizesToLevelMajor :: [[Int]] -> ISL s [[Int]]
tileSizesToLevelMajor [] =
    throwISL "applyScheduleOptimization(Tile): at least one axis is required"
tileSizesToLevelMajor ([] : _) =
    throwISL "applyScheduleOptimization(Tile): at least one tile size per axis is required"
tileSizesToLevelMajor sizesByAxis@(firstAxis : restAxes) = do
    when (any null restAxes) $
        throwISL "applyScheduleOptimization(Tile): at least one tile size per axis is required"
    let levelCount = length firstAxis
    unless (all (\sizes -> length sizes == levelCount) sizesByAxis) $
        throwISL "applyScheduleOptimization(Tile): all axes must have the same level count"
    unless (all (> 0) (concat sizesByAxis)) $
        throwISL "applyScheduleOptimization(Tile): tile sizes must be positive"
    pure (transpose sizesByAxis)
