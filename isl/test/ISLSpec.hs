module ISLSpec (spec) where

import           ISL
import           Test.Hspec

-- Helper to unwrap Right or fail
shouldBeRight :: (Show e) => Either e a -> IO a
shouldBeRight (Right a) = pure a
shouldBeRight (Left e)  = fail $ "Expected Right but got Left: " <> show e

spec :: Spec
spec = do
    describe "ISL Context" $ do
        it "can run ISL monad" $ do
            result <- runISL $ do
                s <- set "{ [i] : 0 <= i < 10 }"
                setToString s
            r <- shouldBeRight result
            r `shouldBe` "{ [i] : 0 <= i <= 9 }"

    describe "ISL Set" $ do
        it "can parse and convert set to string" $ do
            result <- runISL $ do
                s <- set "{ [i] : 0 <= i < 10 }"
                setToString s
            r <- shouldBeRight result
            r `shouldBe` "{ [i] : 0 <= i <= 9 }"

        it "can compute set union" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 5 }"
                s2 <- set "{ [i] : 5 <= i < 10 }"
                u <- s1 \/ s2
                setToString u
            r <- shouldBeRight result
            r `shouldBe` "{ [i] : 0 <= i <= 9 and (i <= 4 or i >= 5) }"

        it "can compute set intersection" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 10 }"
                s2 <- set "{ [i] : 5 <= i < 15 }"
                i <- s1 /\ s2
                setToString i
            r <- shouldBeRight result
            r `shouldBe` "{ [i] : 5 <= i <= 9 }"

        it "can compute set subtraction" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 10 }"
                s2 <- set "{ [i] : 5 <= i < 10 }"
                d <- s1 \\ s2
                setToString d
            r <- shouldBeRight result
            r `shouldBe` "{ [i] : 0 <= i <= 4 }"

        it "can coalesce set" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 5 }"
                s2 <- set "{ [i] : 5 <= i < 10 }"
                u <- s1 \/ s2
                c <- setCoalesce u
                setToString c
            r <- shouldBeRight result
            r `shouldBe` "{ [i] : 0 <= i <= 9 }"

    describe "ISL Union Set" $ do
        it "can parse and convert union set to string" $ do
            result <- runISL $ do
                uset <- unionSet "{ A[i] : 0 <= i < 10; B[j] : 0 <= j < 5 }"
                unionSetToString uset
            r <- shouldBeRight result
            r `shouldBe` "{ A[i] : 0 <= i <= 9; B[j] : 0 <= j <= 4 }"

        it "can compute union set union" $ do
            result <- runISL $ do
                u1 <- unionSet "{ A[i] : 0 <= i < 5 }"
                u2 <- unionSet "{ B[j] : 0 <= j < 3 }"
                combined <- unionSetUnion u1 u2
                unionSetToString combined
            r <- shouldBeRight result
            r `shouldBe` "{ A[i] : 0 <= i <= 4; B[j] : 0 <= j <= 2 }"

    describe "ISL Schedule" $ do
        it "can parse and convert schedule to string" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                scheduleToString sched
            r <- shouldBeRight result
            r `shouldBe` "{ domain: \"{ S[i] : 0 <= i <= 9 }\", child: { schedule: \"[{ S[i] -> [(i)] }]\" } }"

        it "can get schedule domain" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                dom <- scheduleDomain sched
                unionSetToString dom
            r <- shouldBeRight result
            r `shouldBe` "{ S[i] : 0 <= i <= 9 }"

        it "can create schedule from domain" $ do
            result <- runISL $ do
                dom <- unionSet "{ S[i] : 0 <= i < 10 }"
                sched <- scheduleFromDomain dom
                scheduleToString sched
            r <- shouldBeRight result
            r `shouldBe` "{ domain: \"{ S[i] : 0 <= i <= 9 }\" }"
