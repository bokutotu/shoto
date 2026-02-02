module Polyhedral (
    synthesize,
    RawPolyhedralModel (..),
) where

import           ISL                          (AstTree, ISL,
                                               astBuildFromContext,
                                               astBuildNodeFromSchedule,
                                               astNodeToTree)
import           Polyhedral.AnalyzeDependence (analyzeDependences)
import           Polyhedral.Parse             (RawPolyhedralModel (..),
                                               parsePolyhedralModel)
import           Polyhedral.Schedule          (computeSchedule)
import           Polyhedral.Types             (PolyhedralModel (..))

synthesize :: RawPolyhedralModel -> ISL s AstTree
synthesize raw = do
    model@PolyhedralModel{context = ctx, domain = dom} <- parsePolyhedralModel raw
    schedule <- analyzeDependences model >>= computeSchedule ctx dom
    build <- astBuildFromContext ctx
    node <- astBuildNodeFromSchedule build schedule
    astNodeToTree node
