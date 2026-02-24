{-# LANGUAGE OverloadedStrings #-}

module FrontendIRSpec (spec) where

import qualified Data.List.NonEmpty as NE
import           FrontendIR         (Axis (..), Expr (..), FrontendError (..),
                                     IxExpr (..), Program (..),
                                     ReductionOp (..), Stmt (..),
                                     TensorDecl (..), checkProgram, lowerToRaw)
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
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                                , TensorDecl{tensor = "B", shape = ["N", "M"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                ]
                        , stmt =
                            Assign
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
                        , tensors =
                            NE.fromList
                                [TensorDecl{tensor = "A", shape = ["N"]}]
                        , stmt =
                            Assign
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
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                ]
                        , stmt =
                            Assign
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
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                ]
                        , stmt =
                            Assign
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "i"]
                                , rhs = ELoad "A" [IxVar "i", IxVar "i"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrDuplicateIter "i")

        it "fails when duplicate tensor declarations exist" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N"]}
                                , TensorDecl{tensor = "A", shape = ["N"]}
                                ]
                        , stmt =
                            Assign
                                { outputTensor = "A"
                                , outputIndex = [IxVar "i"]
                                , rhs = EConst 0
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrDuplicateTensor "A")

        it "fails when loading from an undeclared tensor" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                ]
                        , tensors =
                            NE.fromList
                                [TensorDecl{tensor = "C", shape = ["N", "M"]}]
                        , stmt =
                            Assign
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "j"]
                                , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrUndeclaredTensor "A")

        it "fails when storing to an undeclared tensor" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [TensorDecl{tensor = "A", shape = ["N"]}]
                        , stmt =
                            Assign
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i"]
                                , rhs = ELoad "A" [IxVar "i"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrUndeclaredTensor "C")

        it "fails when load rank does not match tensor rank" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                ]
                        , stmt =
                            Assign
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "j"]
                                , rhs = ELoad "A" [IxVar "i"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrTensorRankMismatch "A" 2 1)

        it "fails when store rank does not match tensor rank" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                                , TensorDecl{tensor = "C", shape = ["N"]}
                                ]
                        , stmt =
                            Assign
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "j"]
                                , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrTensorRankMismatch "C" 1 2)

        it "fails when tensor shape uses undeclared axis parameter" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "K"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                ]
                        , stmt =
                            Assign
                                { outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "j"]
                                , rhs = EConst 0
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrUnknownTensorShapeParam "A" "K")

        it "lowers a simple reduction program" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "k", extent = "K"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "K"]}
                                , TensorDecl{tensor = "C", shape = ["N"]}
                                ]
                        , stmt =
                            Reduction
                                { reductionOp = ReduceAdd
                                , outputTensor = "C"
                                , outputIndex = [IxVar "i"]
                                , rhs = ELoad "A" [IxVar "i", IxVar "k"]
                                }
                        }

                expected =
                    RawPolyhedralModel
                        { context = "[N,K] -> { : 0 <= N and 0 <= K }"
                        , domain = "[N,K] -> { S[i,k] : 0 <= i < N and 0 <= k < K }"
                        , programOrder = "[N,K] -> { S[i,k] -> [i,k] }"
                        , readAccess = "[N,K] -> { S[i,k] -> A[i,k] }"
                        , writeAccess = "[N,K] -> { S[i,k] -> C[i] }"
                        , reductionDomain = "[N,K] -> { S[i,k] : 0 <= i < N and 0 <= k < K }"
                        , reductionRead = "[N,K] -> { S[i,k] -> C[i] }"
                        , reductionWrite = "[N,K] -> { S[i,k] -> C[i] }"
                        }

            fmap lowerToRaw (checkProgram front) `shouldBe` Right expected

        it "fails when reduction output index is not an ordered loop subsequence" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                , Axis{iter = "k", extent = "K"}
                                ]
                        , tensors =
                            NE.fromList
                                [TensorDecl{tensor = "C", shape = ["K", "N"]}]
                        , stmt =
                            Reduction
                                { reductionOp = ReduceAdd
                                , outputTensor = "C"
                                , outputIndex = [IxVar "k", IxVar "i"]
                                , rhs = EConst 0
                                }
                        }

            fmap lowerToRaw (checkProgram invalid)
                `shouldBe` Left (ErrReductionOutputNotSubsequence ["i", "j", "k"] ["k", "i"])

        it "fails when reduction does not reduce any axis" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                ]
                        , tensors =
                            NE.fromList
                                [TensorDecl{tensor = "C", shape = ["N", "M"]}]
                        , stmt =
                            Reduction
                                { reductionOp = ReduceAdd
                                , outputTensor = "C"
                                , outputIndex = [IxVar "i", IxVar "j"]
                                , rhs = EConst 1
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left ErrReductionRequiresReducedAxis

        it "fails when reduction load index uses undeclared iterator" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "k", extent = "K"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "K"]}
                                , TensorDecl{tensor = "C", shape = ["N"]}
                                ]
                        , stmt =
                            Reduction
                                { reductionOp = ReduceAdd
                                , outputTensor = "C"
                                , outputIndex = [IxVar "i"]
                                , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                }
                        }

            fmap lowerToRaw (checkProgram invalid) `shouldBe` Left (ErrUnknownIndexIter "j")
