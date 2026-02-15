module ISL.ScheduleConstraintsSpec (spec) where

import           ISL
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Schedule Constraints" $ do
        it "can create schedule constraints from domain" $ do
            result <- runISL $ do
                domain <- unionSet "{ S[i] : 0 <= i < 10 }"
                sc <- scheduleConstraintsOnDomain domain
                sched <- scheduleConstraintsComputeSchedule sc
                scheduleDomain sched >>= unionSetToString
            result `shouldSatisfy` isRight
            let Right domStr = result
            domStr `shouldContain` "S["

        it "can compute schedule with validity constraints (dependencies)" $ do
            result <- runISL $ do
                -- Domain: S[i] for i in [0, 9]
                domain <- unionSet "{ S[i] : 0 <= i < 10 }"
                -- Dependency: S[i] must execute before S[i+1]
                deps <- unionMap "{ S[i] -> S[i+1] : 0 <= i < 9 }"

                sc <- scheduleConstraintsOnDomain domain
                sc' <- scheduleConstraintsSetValidity sc deps
                sched <- scheduleConstraintsComputeSchedule sc'

                -- Generate AST
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToC node
            result `shouldSatisfy` isRight
            let Right code = result
            -- Should generate a for loop
            code `shouldContain` "for"

        it "can compute schedule with multiple statements and dependencies" $ do
            result <- runISL $ do
                -- Two statements: S and T
                domain <- unionSet "{ S[i] : 0 <= i < 5; T[i] : 0 <= i < 5 }"
                -- S[i] must execute before T[i]
                deps <- unionMap "{ S[i] -> T[i] : 0 <= i < 5 }"

                sc <- scheduleConstraintsOnDomain domain
                sc' <- scheduleConstraintsSetValidity sc deps
                sched <- scheduleConstraintsComputeSchedule sc'

                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToC node
            result `shouldSatisfy` isRight
            let Right code = result
            code `shouldContain` "S("
            code `shouldContain` "T("

        it "can set proximity constraints" $ do
            result <- runISL $ do
                domain <- unionSet "{ S[i] : 0 <= i < 10 }"
                deps <- unionMap "{ S[i] -> S[i+1] : 0 <= i < 9 }"
                -- Proximity: try to keep S[i] and S[i+1] close in time
                proximity <- unionMap "{ S[i] -> S[i+1] : 0 <= i < 9 }"

                sc <- scheduleConstraintsOnDomain domain
                sc' <- scheduleConstraintsSetValidity sc deps
                sc'' <- scheduleConstraintsSetProximity sc' proximity
                sched <- scheduleConstraintsComputeSchedule sc''

                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToC node
            result `shouldSatisfy` isRight

        it "can set coincidence constraints" $ do
            result <- runISL $ do
                domain <- unionSet "{ S[i,j] : 0 <= i < 4 and 0 <= j < 4 }"
                -- No dependencies - can parallelize
                deps <- unionMap "{ }"
                -- Coincidence: try to execute S[i,*] together (parallelize outer loop)
                coincidence <- unionMap "{ S[i,j] -> S[i,j'] : 0 <= i < 4 and 0 <= j,j' < 4 }"

                sc <- scheduleConstraintsOnDomain domain
                sc' <- scheduleConstraintsSetValidity sc deps
                sc'' <- scheduleConstraintsSetCoincidence sc' coincidence
                sched <- scheduleConstraintsComputeSchedule sc''

                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToC node
            result `shouldSatisfy` isRight

isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _ = False
