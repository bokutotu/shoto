module Shoto (compile) where

import           ISL (IslError, runISL, unionAccessInfoComputeFlow,
                      unionAccessInfoFromSink, unionAccessInfoSetMustSource,
                      unionAccessInfoSetScheduleMap, unionFlowGetMustDependence,
                      unionMap, unionMapIntersectDomain, unionMapToString,
                      unionSet)

compile :: String -> String -> String -> String -> IO (Either IslError String)
compile domainStr writeStr reedStr scheduleStr = runISL $ do
    domain <- unionSet domainStr
    write <- unionMap writeStr
    reed <- unionMap reedStr
    schedule <- unionMap scheduleStr

    -- Intersect with domain
    write' <- unionMapIntersectDomain write domain
    reed' <- unionMapIntersectDomain reed domain
    schedule' <- unionMapIntersectDomain schedule domain

    -- Compute dependence
    accessInfo <- unionAccessInfoFromSink reed'
    accessInfo' <- unionAccessInfoSetMustSource accessInfo write'
    accessInfo'' <- unionAccessInfoSetScheduleMap accessInfo' schedule'
    flow <- unionAccessInfoComputeFlow accessInfo''
    dep <- unionFlowGetMustDependence flow

    unionMapToString dep
