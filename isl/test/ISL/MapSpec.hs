module ISL.MapSpec (spec) where

import           ISL
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Map" $ do
        it "can parse and convert map to string" $ do
            result <- runISL $ do
                m <- imap "{ [i] -> [j] : 0 <= i < 10 and j = i + 1 }"
                mapToString m
            result `shouldBe` Right "{ [i] -> [j = 1 + i] : 0 <= i <= 9 }"

        it "can compute map union" $ do
            result <- runISL $ do
                m1 <- imap "{ [i] -> [j] : 0 <= i < 5 and j = i }"
                m2 <- imap "{ [i] -> [j] : 5 <= i < 10 and j = i }"
                u <- mapUnion m1 m2
                mapToString u
            result `shouldBe` Right "{ [i] -> [j = i] : 0 <= i <= 9 and (i <= 4 or i >= 5) }"

        it "can compute map intersection" $ do
            result <- runISL $ do
                m1 <- imap "{ [i] -> [j] : 0 <= i < 10 and j = i }"
                m2 <- imap "{ [i] -> [j] : 5 <= i < 15 and j = i }"
                i <- mapIntersect m1 m2
                mapToString i
            result `shouldBe` Right "{ [i] -> [j = i] : 5 <= i <= 9 }"

        it "can compute map subtraction" $ do
            result <- runISL $ do
                m1 <- imap "{ [i] -> [j] : 0 <= i < 10 and j = i }"
                m2 <- imap "{ [i] -> [j] : 5 <= i < 10 and j = i }"
                d <- mapSubtract m1 m2
                mapToString d
            result `shouldBe` Right "{ [i] -> [j = i] : 0 <= i <= 4 }"

        it "can coalesce map" $ do
            result <- runISL $ do
                m1 <- imap "{ [i] -> [j] : 0 <= i < 5 and j = i }"
                m2 <- imap "{ [i] -> [j] : 5 <= i < 10 and j = i }"
                u <- mapUnion m1 m2
                c <- mapCoalesce u
                mapToString c
            result `shouldBe` Right "{ [i] -> [j = i] : 0 <= i <= 9 }"

        it "can get map domain" $ do
            result <- runISL $ do
                m <- imap "{ [i] -> [j] : 0 <= i < 10 and j = i + 1 }"
                d <- mapDomain m
                setToString d
            result `shouldBe` Right "{ [i] : 0 <= i <= 9 }"

        it "can get map range" $ do
            result <- runISL $ do
                m <- imap "{ [i] -> [j] : 0 <= i < 10 and j = i + 1 }"
                r <- mapRange m
                setToString r
            result `shouldBe` Right "{ [j] : 0 < j <= 10 }"

        it "can reverse map" $ do
            result <- runISL $ do
                m <- imap "{ [i] -> [j] : 0 <= i < 10 and j = 2*i }"
                r <- mapReverse m
                mapToString r
            result `shouldBe` Right "{ [j] -> [i] : 2i = j and 0 <= j <= 18 }"

        it "can apply range" $ do
            result <- runISL $ do
                m1 <- imap "{ [i] -> [j] : 0 <= i < 10 and j = i }"
                m2 <- imap "{ [i] -> [j] : j = i + 1 }"
                r <- mapApplyRange m1 m2
                mapToString r
            result `shouldBe` Right "{ [i] -> [j = 1 + i] : 0 <= i <= 9 }"

    describe "ISL UnionMap" $ do
        it "can parse and convert union map to string" $ do
            result <- runISL $ do
                m <- unionMap "{ A[i] -> B[j] : 0 <= i < 10 and j = i }"
                unionMapToString m
            result `shouldBe` Right "{ A[i] -> B[j = i] : 0 <= i <= 9 }"

        it "can compute union map union" $ do
            result <- runISL $ do
                m1 <- unionMap "{ A[i] -> B[j] : 0 <= i < 5 and j = i }"
                m2 <- unionMap "{ C[i] -> D[j] : 0 <= i < 5 and j = i }"
                u <- unionMapUnion m1 m2
                unionMapToString u
            case result of
                Right s -> s `shouldContain` "A[i] -> B[j = i]"
                Left e  -> expectationFailure $ show e

        it "can get union map domain" $ do
            result <- runISL $ do
                m <- unionMap "{ A[i] -> B[j] : 0 <= i < 10 and j = i }"
                d <- unionMapDomain m
                unionSetToString d
            result `shouldBe` Right "{ A[i] : 0 <= i <= 9 }"

        it "can get union map range" $ do
            result <- runISL $ do
                m <- unionMap "{ A[i] -> B[j] : 0 <= i < 10 and j = i }"
                r <- unionMapRange m
                unionSetToString r
            result `shouldBe` Right "{ B[j] : 0 <= j <= 9 }"

        it "can reverse union map" $ do
            result <- runISL $ do
                m <- unionMap "{ A[i] -> B[j] : 0 <= i < 10 and j = i }"
                r <- unionMapReverse m
                unionMapToString r
            result `shouldBe` Right "{ B[j] -> A[i = j] : 0 <= j <= 9 }"
