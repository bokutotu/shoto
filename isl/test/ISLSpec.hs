module ISLSpec (spec) where

import ISL
import Test.Hspec

spec :: Spec
spec = do
    describe "ISL Context" $ do
        it "can run ISL monad" $ do
            result <- runISL $ do
                s <- set "{ [i] : 0 <= i < 10 }"
                setToString s
            result `shouldContain` "i"

    describe "ISL Set (Domain)" $ do
        it "can parse and convert set to string" $ do
            result <- runISL $ do
                s <- set "{ [i] : 0 <= i < 10 }"
                setToString s
            result `shouldContain` "i"

    describe "ISL Union Set" $ do
        it "can parse and convert union set to string" $ do
            result <- runISL $ do
                uset <- unionSet "{ A[i] : 0 <= i < 10; B[j] : 0 <= j < 5 }"
                unionSetToString uset
            result `shouldContain` "A"

    describe "ISL Schedule" $ do
        it "can parse and convert schedule to string" $ do
            result <- runISL $ do
                sched <-
                    schedule "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                scheduleToString sched
            result `shouldContain` "S"
