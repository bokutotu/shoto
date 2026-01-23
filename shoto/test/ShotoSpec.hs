module ShotoSpec (spec) where

import           ISL        (AstTree (..))
import           Shoto      (compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "can compute RAW dependence for simple copy" $ do
            let domain = "[N] -> { S[i] : 0 <= i < N }"
                write = "{ S[i] -> A[i] }"
                reed = "{ S[i] -> B[i] }"
                schedule = "{ S[i] -> [i] }"
                params = ["N"]

            result <- compile domain write reed schedule params
            result `shouldBe` Right AstError
