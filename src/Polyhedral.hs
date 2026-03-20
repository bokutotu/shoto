{-# LANGUAGE LambdaCase #-}

module Polyhedral (
    synthesize,
    RawPolyhedralModel (..),
    ScheduleOptimization (..),
) where

import           Control.Monad.Except         (catchError, throwError)
import           Polyhedral.AnalyzeDependence (analyzeDependences)
import           Polyhedral.Error             (PolyhedralError (..))
import           Polyhedral.Internal          (AstTree, ISL,
                                               astBuildFromContext,
                                               astBuildNodeFromSchedule,
                                               astNodeToTree)
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
    build <- withAstError (astBuildFromContext ctx)
    node <- withAstError (astBuildNodeFromSchedule build optimizedSchedule)
    withAstError (astNodeToTree node)

withAstError :: ISL s a -> ISL s a
withAstError action =
    catchError
        action
        ( \case
            InternalIslError islErr ->
                throwError (PolyhedralAstError (Just islErr))
            other -> throwError other
        )
