{-# LANGUAGE OverloadedStrings #-}

module FrontendIRSpec (spec) where

import qualified Data.List.NonEmpty as NE
import           FrontendIR         (Axis (..), Expr (..), FrontendError (..),
                                     IxExpr (..), Program (..), Stmt (..),
                                     checkProgram, lowerToRaw)
import           Polyhedral         (RawPolyhedralModel (..))
import           Test.Hspec

spec :: Spec
spec = do
    describe "FrontendIR lowering" $ do
        it "lowers a simple 2D point-wise add program" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                ]
                        , stmt =
                            Stmt
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "j"]
                                , rhs =
                                    EAdd
                                        (ELoad "A" [IxVar "i", IxVar "j"])
                                        (ELoad "B" [IxVar "i", IxVar "j"])
                                }
                        }

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

            fmap lowerToRaw (checkProgram front) `shouldBe` Right expected

        it "lowers a constant write with empty read access" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , stmt =
                            Stmt
                                { outputTensor = "A"
                                , outputIndex = [IxVar "i"]
                                , rhs = EConst 42
                                }
                        }

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

            fmap lowerToRaw (checkProgram front) `shouldBe` Right expected

        it "fails when store indices do not match loop axes" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                ]
                        , stmt =
                            Stmt
                                { outputTensor = "C"
                                , outputIndex = [IxVar "j", IxVar "i"]
                                , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrStoreIndexMismatch ["i", "j"] ["j", "i"])

        it "fails when duplicate iterators are declared" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "i", extent = "M"}
                                ]
                        , stmt =
                            Stmt
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "i"]
                                , rhs = ELoad "A" [IxVar "i", IxVar "i"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrDuplicateIter "i")
