module Polyhedral (
    synthesize,
    RawPolyhedralModel (..),
    ScheduleOptimization (..),
) where

import           ISL                          (AstTree, ISL,
                                               astBuildFromContext,
                                               astBuildNodeFromSchedule,
                                               astNodeToTree)
import           Polyhedral.AnalyzeDependence (analyzeDependences)
import           Polyhedral.Optimize          (ScheduleOptimization (..),
                                               applyScheduleOptimizations)
import           Polyhedral.Parse             (RawPolyhedralModel (..),
                                               parsePolyhedralModel)
import           Polyhedral.Schedule          (computeSchedule)
import           Polyhedral.Types             (PolyhedralModel (..))

synthesize :: [ScheduleOptimization] -> RawPolyhedralModel -> ISL s AstTree
synthesize optimizations raw = do
    model@PolyhedralModel{context = ctx, domain = dom} <- parsePolyhedralModel raw
    schedule <- analyzeDependences model >>= computeSchedule ctx dom
    optimizedSchedule <- applyScheduleOptimizations optimizations schedule
    build <- astBuildFromContext ctx
    node <- astBuildNodeFromSchedule build optimizedSchedule
    astNodeToTree node
