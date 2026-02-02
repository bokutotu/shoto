module ISL.SetSpec (spec) where

import           ISL
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Set" $ do
        it "can parse and convert set to string" $ do
            result <- runISL $ do
                s <- set "{ [i] : 0 <= i < 10 }"
                setToString s
            result `shouldBe` Right "{ [i] : 0 <= i <= 9 }"

        it "can compute set union" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 5 }"
                s2 <- set "{ [i] : 5 <= i < 10 }"
                u <- setUnion s1 s2
                setToString u
            result `shouldBe` Right "{ [i] : 0 <= i <= 9 and (i <= 4 or i >= 5) }"

        it "can compute set intersection" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 10 }"
                s2 <- set "{ [i] : 5 <= i < 15 }"
                i <- setIntersect s1 s2
                setToString i
            result `shouldBe` Right "{ [i] : 5 <= i <= 9 }"

        it "can compute set subtraction" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 10 }"
                s2 <- set "{ [i] : 5 <= i < 10 }"
                d <- setSubtract s1 s2
                setToString d
            result `shouldBe` Right "{ [i] : 0 <= i <= 4 }"

        it "can coalesce set" $ do
            result <- runISL $ do
                s1 <- set "{ [i] : 0 <= i < 5 }"
                s2 <- set "{ [i] : 5 <= i < 10 }"
                u <- setUnion s1 s2
                c <- setCoalesce u
                setToString c
            result `shouldBe` Right "{ [i] : 0 <= i <= 9 }"

    describe "ISL UnionSet" $ do
        it "returns True for empty union set" $ do
            result <- runISL $ do
                s <- unionSet "{ }"
                unionSetIsEmpty s
            result `shouldBe` Right True

        it "returns False for non-empty union set" $ do
            result <- runISL $ do
                s <- unionSet "{ A[i] : 0 <= i < 10 }"
                unionSetIsEmpty s
            result `shouldBe` Right False
