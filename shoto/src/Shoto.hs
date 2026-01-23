module Shoto (compile) where

import           Control.Monad (unless)
import           Data.List     (intercalate)
import           ISL           (AstTree, ISL, IslError, UnionMap,
                                astBuildFromContext, astBuildNodeFromSchedule,
                                astNodeToTree, runISL,
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
computeDeps source sink schedule = do
    accessInfo <- unionAccessInfoFromSink sink
    accessInfo' <- unionAccessInfoSetMustSource accessInfo source
    accessInfo'' <- unionAccessInfoSetScheduleMap accessInfo' schedule
    flow <- unionAccessInfoComputeFlow accessInfo''
    unionFlowGetMustDependence flow

compile :: String -> String -> String -> String -> [String] -> IO (Either IslError AstTree)
compile domainStr writeStr reedStr scheduleStr params = runISL $ do
    domain <- unionSet domainStr
    write <- unionMap writeStr
    reed <- unionMap reedStr
    schedule <- unionMap scheduleStr

    -- Intersect with domain
    write' <- unionMapIntersectDomain write domain
    reed' <- unionMapIntersectDomain reed domain
    schedule' <- unionMapIntersectDomain schedule domain

    -- Compute dependence
    rawDep <- computeDeps write' reed' schedule'
    wawDep <- computeDeps write' write' schedule'
    warDep <- computeDeps reed' write' schedule'

    dep <- unionMapUnion rawDep warDep >>= unionMapUnion wawDep
    order <- unionMapLexLt schedule' schedule'
    violations <- unionMapSubtract dep order
    ok <- unionMapIsEmpty violations
    unless ok $ throwISL "Found dependences that violate the schedule"

    validDep <- unionMapUnion dep order
    sc <- scheduleConstraintsOnDomain domain
    sc' <- scheduleConstraintsSetValidity sc validDep
    sc'' <- scheduleConstraintsSetProximity sc' rawDep

    newSchedule <- scheduleConstraintsComputeSchedule sc''

    paramSet <- set $ "[" ++ intercalate "," params ++ "]" ++ "-> { : }"
    build <- astBuildFromContext paramSet
    ast <- astBuildNodeFromSchedule build newSchedule
    astNodeToTree ast
