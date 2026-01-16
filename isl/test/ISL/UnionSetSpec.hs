module ISL.UnionSetSpec (spec) where

import           ISL
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Union Set" $ do
        it "can parse and convert union set to string" $ do
            result <- runISL $ do
                uset <- unionSet "{ A[i] : 0 <= i < 10; B[j] : 0 <= j < 5 }"
                unionSetToString uset
            case result of
                Left err -> expectationFailure $ show err
                Right str -> do
                    str `shouldContain` "A[i]"
                    str `shouldContain` "B[j]"

        it "can compute union set union" $ do
            result <- runISL $ do
                u1 <- unionSet "{ A[i] : 0 <= i < 5 }"
                u2 <- unionSet "{ B[j] : 0 <= j < 3 }"
                combined <- unionSetUnion u1 u2
                unionSetToString combined
            case result of
                Left err -> expectationFailure $ show err
                Right str -> do
                    str `shouldContain` "A[i]"
                    str `shouldContain` "B[j]"
