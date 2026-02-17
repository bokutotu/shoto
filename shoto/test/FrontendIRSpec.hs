{-# LANGUAGE OverloadedStrings #-}

module FrontendIRSpec (spec) where

import           FrontendIR (FrontendError (..), axis, iconst, load, lowerToRaw,
                             program, store, (.+.))
import           Polyhedral (RawPolyhedralModel (..))
import           Test.Hspec

spec :: Spec
spec = do
    describe "FrontendIR lowering" $ do
        it "lowers a simple 2D point-wise add program" $ do
            let front =
                    program
                        [ axis "i" "N"
                        , axis "j" "M"
                        ]
                        (store "C" ["i", "j"] (load "A" ["i", "j"] .+. load "B" ["i", "j"]))

                expected =
                    RawPolyhedralModel
                        { context = "[N,M] -> { : 0 <= N and 0 <= M }"
                        , domain = "[N,M] -> { S[i,j] : 0 <= i < N and 0 <= j < M }"
                        , programOrder = "[N,M] -> { S[i,j] -> [i,j] }"
                        , readAccess = "[N,M] -> { S[i,j] -> A[i,j]; S[i,j] -> B[i,j] }"
                        , writeAccess = "[N,M] -> { S[i,j] -> C[i,j] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

            lowerToRaw front `shouldBe` Right expected

        it "lowers a constant write with empty read access" $ do
            let front =
                    program
                        [axis "i" "N"]
                        (store "A" ["i"] (iconst 42))

                expected =
                    RawPolyhedralModel
                        { context = "[N] -> { : 0 <= N }"
                        , domain = "[N] -> { S[i] : 0 <= i < N }"
                        , programOrder = "[N] -> { S[i] -> [i] }"
                        , readAccess = "{ }"
                        , writeAccess = "[N] -> { S[i] -> A[i] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

            lowerToRaw front `shouldBe` Right expected

        it "fails when store indices do not match loop axes" $ do
            let invalid =
                    program
                        [ axis "i" "N"
                        , axis "j" "M"
                        ]
                        (store "C" ["j", "i"] (load "A" ["i", "j"]))

            lowerToRaw invalid `shouldBe` Left (ErrStoreIndexMismatch ["i", "j"] ["j", "i"])

        it "fails when duplicate iterators are declared" $ do
            let invalid =
                    program
                        [ axis "i" "N"
                        , axis "i" "M"
                        ]
                        (store "C" ["i", "i"] (load "A" ["i", "i"]))

            lowerToRaw invalid `shouldBe` Left (ErrDuplicateIter "i")
