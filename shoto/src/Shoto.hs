{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot   #-}
{-# LANGUAGE RecordWildCards       #-}

module Shoto (compile, CompilerArgs (..)) where

import           Control.Monad (unless)
import           Data.List     (intercalate)
import           ISL           (AstTree, ISL, IslError, Schedule, Set, UnionMap,
                                UnionSet, astBuildFromContext,
                                astBuildNodeFromSchedule, astNodeToTree, runISL,
                                scheduleConstraintsComputeSchedule,
                                scheduleConstraintsOnDomain,
                                scheduleConstraintsSetProximity,
                                scheduleConstraintsSetValidity, set, throwISL,
                                unionAccessInfoComputeFlow,
                                unionAccessInfoFromSink,
                                unionAccessInfoSetMustSource,
                                unionAccessInfoSetScheduleMap,
                                unionFlowGetMustDependence, unionMap,
                                unionMapIntersectDomain, unionMapIsEmpty,
                                unionMapLexLt, unionMapSubtract, unionMapUnion,
                                unionSet)

computeDeps :: UnionMap s -> UnionMap s -> UnionMap s -> ISL s (UnionMap s)
computeDeps source sink schedule =
    unionAccessInfoFromSink sink
        >>= flip unionAccessInfoSetMustSource source
        >>= flip unionAccessInfoSetScheduleMap schedule
        >>= unionAccessInfoComputeFlow
        >>= unionFlowGetMustDependence

data Deps s = Deps
    { raw     :: UnionMap s
    , waw     :: UnionMap s
    , war     :: UnionMap s
    , allDeps :: UnionMap s
    }

analyzeDeps :: UnionMap s -> UnionMap s -> UnionMap s -> ISL s (Deps s)
analyzeDeps write reed schedule = do
    rawDep <- computeDeps write reed schedule
    wawDep <- computeDeps write write schedule
    warDep <- computeDeps reed write schedule
    allDeps <- unionMapUnion rawDep wawDep >>= unionMapUnion warDep
    pure $ Deps rawDep wawDep warDep allDeps

validateSchedule :: UnionMap s -> UnionMap s -> ISL s ()
validateSchedule order allDeps = do
    violations <- unionMapSubtract allDeps order
    ok <- unionMapIsEmpty violations
    unless ok $ throwISL "Found dependences that violate the schedule"

data CompilerArgs = CompilerArgs
    { domain   :: String
    , write    :: String
    , reed     :: String
    , schedule :: String
    , params   :: [String]
    }

data ParsedArgs s = ParserArgs
    { domain   :: UnionSet s
    , write    :: UnionMap s
    , reed     :: UnionMap s
    , schedule :: UnionMap s
    , params   :: Set s
    }

parseArgs :: CompilerArgs -> ISL s (ParsedArgs s)
parseArgs args = do
    domain <- unionSet args.domain
    write <- unionMap args.write >>= flip unionMapIntersectDomain domain
    reed <- unionMap args.reed >>= flip unionMapIntersectDomain domain
    schedule <- unionMap args.schedule >>= flip unionMapIntersectDomain domain
    params <- set $ "[" ++ intercalate "," args.params ++ "]" ++ "-> { : }"
    pure $ ParserArgs domain write reed schedule params

computeSchedule :: UnionMap s -> UnionMap s -> UnionMap s -> UnionSet s -> ISL s (Schedule s)
computeSchedule deps rawDep order domain = do
    validDep <- unionMapUnion deps order
    sc <- scheduleConstraintsOnDomain domain
    sc' <- scheduleConstraintsSetValidity sc validDep
    sc'' <- scheduleConstraintsSetProximity sc' rawDep
    scheduleConstraintsComputeSchedule sc''

buildAst :: Set s -> Schedule s -> ISL s AstTree
buildAst params schedule = do
    build <- astBuildFromContext params
    ast <- astBuildNodeFromSchedule build schedule
    astNodeToTree ast

compile :: CompilerArgs -> IO (Either IslError AstTree)
compile args = runISL $ do
    ParserArgs{..} <- parseArgs args
    deps <- analyzeDeps write reed schedule
    order <- unionMapLexLt schedule schedule
    validateSchedule order deps.allDeps
    newSchedule <- computeSchedule deps.allDeps deps.raw order domain
    buildAst params newSchedule
