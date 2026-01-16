module ISL.ScheduleTreeSpec (spec) where

import           ISL
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Schedule Tree" $ do
        it "can get schedule tree from domain-only schedule" $ do
            result <- runISL $ do
                dom <- unionSet "{ S[i] : 0 <= i < 10 }"
                sched <- scheduleFromDomain dom
                scheduleTree sched
            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        TreeDomain _ [TreeLeaf] -> pure ()
                        _ -> expectationFailure $ "Unexpected tree: " ++ show tree

        it "can get schedule tree with band node" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                scheduleTree sched
            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        TreeDomain _ [TreeBand info [TreeLeaf]] -> do
                            bandMembers info `shouldBe` 1
                        _ -> expectationFailure $ "Unexpected tree: " ++ show tree

        it "can get schedule tree with sequence node" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ A[i] : 0 <= i < 5; B[j] : 0 <= j < 3 }\", child: { sequence: [ { filter: \"{ A[i] }\" }, { filter: \"{ B[j] }\" } ] } }"
                scheduleTree sched
            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        TreeDomain _ [TreeSequence [TreeFilter _ [TreeLeaf], TreeFilter _ [TreeLeaf]]] ->
                            pure ()
                        _ -> expectationFailure $ "Unexpected tree: " ++ show tree
