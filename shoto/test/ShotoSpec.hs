module ShotoSpec (spec) where

import           Shoto      (compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "can compute RAW dependence for simple copy" $ do
            -- S: C[i] = A[i] + B[i]  for i = 0..9
            -- T: D[i] = C[i-1]       for i = 1..9
            -- RAW: T reads C[i-1] which S writes at C[i]
            result <-
                compile
                    "{ S[i] : 0 <= i <= 9; T[i] : 1 <= i <= 9 }"
                    "{ S[i] -> C[i] }"
                    "{ T[i] -> C[i-1] }"
                    "{ S[i] -> [0,i]; T[i] -> [1,i] }"
            result `shouldBe` Right "{ S[i] -> T[i' = 1 + i] : 0 <= i <= 8 }"

        it "can detect no dependence when access patterns are disjoint" $ do
            -- S writes to A, T reads from B -> no dependence
            result <-
                compile
                    "{ S[i] : 0 <= i <= 4; T[i] : 0 <= i <= 4 }"
                    "{ S[i] -> A[i] }"
                    "{ T[i] -> B[i] }"
                    "{ S[i] -> [0,i]; T[i] -> [1,i] }"
            result `shouldBe` Right "{  }"

        it "can compute dependence for matrix multiplication" $ do
            -- C[i,j] += A[i,k] * B[k,j]
            -- Self-dependence on C[i,j] across k iterations
            result <-
                compile
                    "{ S[i,j,k] : 0 <= i,j,k <= 2 }"
                    "{ S[i,j,k] -> C[i,j] }"
                    "{ S[i,j,k] -> C[i,j] }"
                    "{ S[i,j,k] -> [i,j,k] }"
            result
                `shouldBe` Right
                    "{ S[i, j, k] -> S[i' = i, j' = j, k' = 1 + k] : 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 1 }"
