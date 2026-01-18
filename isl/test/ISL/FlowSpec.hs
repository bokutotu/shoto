module ISL.FlowSpec (spec) where

import           ISL
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Flow" $ do
        it "can compute RAW dependence" $ do
            -- S: C[i] = A[i] + B[i]  for i = 0..9
            -- T: D[i] = C[i-1]       for i = 1..9
            -- RAW: T reads C[i-1] which S writes at C[i], so T[i] depends on S[i-1]
            result <- runISL $ do
                sink <- unionMap "{ T[i] -> C[i-1] : 1 <= i <= 9 }"
                writes <- unionMap "{ S[i] -> C[i] : 0 <= i <= 9 }"
                sched <- unionMap "{ S[i] -> [0,i]; T[i] -> [1,i] }"
                accessInfo <- unionAccessInfoFromSink sink
                accessInfo' <- unionAccessInfoSetMustSource accessInfo writes
                accessInfo'' <- unionAccessInfoSetScheduleMap accessInfo' sched
                flow <- unionAccessInfoComputeFlow accessInfo''
                dep <- unionFlowGetMustDependence flow
                unionMapToString dep
            result `shouldBe` Right "{ S[i] -> T[i' = 1 + i] : 0 <= i <= 8 }"

        it "can compute RAW dependence for matrix multiplication" $ do
            -- C[i,j] += A[i,k] * B[k,j]
            -- RAW: C[i,j] depends on previous C[i,j]
            result <- runISL $ do
                sink <- unionMap "{ S[i,j,k] -> C[i,j] }"
                writes <- unionMap "{ S[i,j,k] -> C[i,j] }"
                sched <- unionMap "{ S[i,j,k] -> [i,j,k] }"
                accessInfo <- unionAccessInfoFromSink sink
                accessInfo' <- unionAccessInfoSetMustSource accessInfo writes
                accessInfo'' <- unionAccessInfoSetScheduleMap accessInfo' sched
                flow <- unionAccessInfoComputeFlow accessInfo''
                dep <- unionFlowGetMustDependence flow
                unionMapToString dep
            result `shouldBe` Right "{ S[i, j, k] -> S[i' = i, j' = j, k' = 1 + k] }"

        it "can detect no dependence when access patterns are disjoint" $ do
            -- S: A[i] = ...    for i = 0..4
            -- T: B[i] = ...    for i = 0..4
            -- No RAW dependence because they access different arrays
            result <- runISL $ do
                sink <- unionMap "{ T[i] -> B[i] : 0 <= i <= 4 }"
                writes <- unionMap "{ S[i] -> A[i] : 0 <= i <= 4 }"
                sched <- unionMap "{ S[i] -> [0,i]; T[i] -> [1,i] }"
                accessInfo <- unionAccessInfoFromSink sink
                accessInfo' <- unionAccessInfoSetMustSource accessInfo writes
                accessInfo'' <- unionAccessInfoSetScheduleMap accessInfo' sched
                flow <- unionAccessInfoComputeFlow accessInfo''
                dep <- unionFlowGetMustDependence flow
                unionMapToString dep
            result `shouldBe` Right "{  }"
