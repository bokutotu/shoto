module ISL.ScheduleSpec (spec) where

import ISL
import Test.Hspec

spec :: Spec
spec = do
    describe "ISL Schedule" $ do
        it "can parse and convert schedule to string" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                scheduleToString sched
            result `shouldBe` Right "{ domain: \"{ S[i] : 0 <= i <= 9 }\", child: { schedule: \"[{ S[i] -> [(i)] }]\" } }"

        it "can get schedule domain" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                dom <- scheduleDomain sched
                unionSetToString dom
            result `shouldBe` Right "{ S[i] : 0 <= i <= 9 }"

        it "can create schedule from domain" $ do
            result <- runISL $ do
                dom <- unionSet "{ S[i] : 0 <= i < 10 }"
                sched <- scheduleFromDomain dom
                scheduleToString sched
            result `shouldBe` Right "{ domain: \"{ S[i] : 0 <= i <= 9 }\" }"
