{-# LANGUAGE OverloadedStrings #-}

module FrontendIRSpec (spec) where

import qualified Data.List.NonEmpty as NE
import           FrontendIR         (Axis (..), Expr (..), FrontendError (..),
                                     IxExpr (..), Program (..),
                                     ReductionOp (..), Stmt (..),
                                     TensorDecl (..), lowerProgram)
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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs =
                                        EAdd
                                            (ELoad "A" [IxVar "i", IxVar "j"])
                                            (ELoad "B" [IxVar "i", IxVar "j"])
                                    }
                                ]
                        }

                expected =
                    RawPolyhedralModel
                        { context = "[N,M] -> { : 0 <= N and 0 <= M }"
                        , domain = "[N,M] -> { S0[i,j] : 0 <= i < N and 0 <= j < M }"
                        , programOrder = "[N,M] -> { S0[i,j] -> [0,i,j] }"
                        , readAccess = "[N,M] -> { S0[i,j] -> A[i,j]; S0[i,j] -> B[i,j] }"
                        , writeAccess = "[N,M] -> { S0[i,j] -> C[i,j] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

            lowerProgram front `shouldBe` Right expected

        it "lowers a constant write with empty read access" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [TensorDecl{tensor = "A", shape = ["N"]}]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "A"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = EConst 42
                                    }
                                ]
                        }

                expected =
                    RawPolyhedralModel
                        { context = "[N] -> { : 0 <= N }"
                        , domain = "[N] -> { S0[i] : 0 <= i < N }"
                        , programOrder = "[N] -> { S0[i] -> [0,i] }"
                        , readAccess = "{ }"
                        , writeAccess = "[N] -> { S0[i] -> A[i] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

            lowerProgram front `shouldBe` Right expected

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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "j", IxVar "i"]
                                    , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrStoreIndexMismatch ["i", "j"] ["j", "i"])

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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "i"]
                                    , rhs = ELoad "A" [IxVar "i", IxVar "i"]
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrDuplicateIter "i")

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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "A"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = EConst 0
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrDuplicateTensor "A")

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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrUndeclaredTensor "A")

        it "fails when storing to an undeclared tensor" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [TensorDecl{tensor = "A", shape = ["N"]}]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = ELoad "A" [IxVar "i"]
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrUndeclaredTensor "C")

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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = ELoad "A" [IxVar "i"]
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrTensorRankMismatch "A" 2 1)

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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrTensorRankMismatch "C" 1 2)

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
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = EConst 0
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrUnknownTensorShapeParam "A" "K")

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
                        , stmts =
                            NE.fromList
                                [ Reduction
                                    { reductionOp = ReduceAdd
                                    , outputTensor = "C"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = ELoad "A" [IxVar "i", IxVar "k"]
                                    }
                                ]
                        }

                expected =
                    RawPolyhedralModel
                        { context = "[N,K] -> { : 0 <= N and 0 <= K }"
                        , domain = "[N,K] -> { S0[i,k] : 0 <= i < N and 0 <= k < K }"
                        , programOrder = "[N,K] -> { S0[i,k] -> [0,i,k] }"
                        , readAccess = "[N,K] -> { S0[i,k] -> A[i,k] }"
                        , writeAccess = "[N,K] -> { S0[i,k] -> C[i] }"
                        , reductionDomain = "[N,K] -> { S0[i,k] : 0 <= i < N and 0 <= k < K }"
                        , reductionRead = "[N,K] -> { S0[i,k] -> C[i] }"
                        , reductionWrite = "[N,K] -> { S0[i,k] -> C[i] }"
                        }

            lowerProgram front `shouldBe` Right expected

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
                        , stmts =
                            NE.fromList
                                [ Reduction
                                    { reductionOp = ReduceAdd
                                    , outputTensor = "C"
                                    , outputIndex = [IxVar "k", IxVar "i"]
                                    , rhs = EConst 0
                                    }
                                ]
                        }

            lowerProgram invalid
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
                        , stmts =
                            NE.fromList
                                [ Reduction
                                    { reductionOp = ReduceAdd
                                    , outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = EConst 1
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left ErrReductionRequiresReducedAxis

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
                        , stmts =
                            NE.fromList
                                [ Reduction
                                    { reductionOp = ReduceAdd
                                    , outputTensor = "C"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = ELoad "A" [IxVar "i", IxVar "j"]
                                    }
                                ]
                        }

            lowerProgram invalid `shouldBe` Left (ErrUnknownIndexIter "j")

        it "lowers fused gemm + add + relu with mixed reduction and point-wise statements" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                , Axis{iter = "k", extent = "K"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "K"]}
                                , TensorDecl{tensor = "B", shape = ["K", "M"]}
                                , TensorDecl{tensor = "Bias", shape = ["N", "M"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                , TensorDecl{tensor = "Y", shape = ["N", "M"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = EConst 0
                                    }
                                , Reduction
                                    { reductionOp = ReduceAdd
                                    , outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs =
                                        EMul
                                            (ELoad "A" [IxVar "i", IxVar "k"])
                                            (ELoad "B" [IxVar "k", IxVar "j"])
                                    }
                                , Assign
                                    { outputTensor = "Y"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs =
                                        EAdd
                                            (ELoad "C" [IxVar "i", IxVar "j"])
                                            (ELoad "Bias" [IxVar "i", IxVar "j"])
                                    }
                                ]
                        }

                expected =
                    RawPolyhedralModel
                        { context = "[N,M,K] -> { : 0 <= N and 0 <= M and 0 <= K }"
                        , domain =
                            "[N,M,K] -> { S0[i,j] : 0 <= i < N and 0 <= j < M; S1[i,j,k] : 0 <= i < N and 0 <= j < M and 0 <= k < K; S2[i,j] : 0 <= i < N and 0 <= j < M }"
                        , programOrder =
                            "[N,M,K] -> { S0[i,j] -> [0,i,j,0]; S1[i,j,k] -> [1,i,j,k]; S2[i,j] -> [2,i,j,0] }"
                        , readAccess =
                            "[N,M,K] -> { S1[i,j,k] -> A[i,k]; S1[i,j,k] -> B[k,j]; S2[i,j] -> C[i,j]; S2[i,j] -> Bias[i,j] }"
                        , writeAccess =
                            "[N,M,K] -> { S0[i,j] -> C[i,j]; S1[i,j,k] -> C[i,j]; S2[i,j] -> Y[i,j] }"
                        , reductionDomain = "[N,M,K] -> { S1[i,j,k] : 0 <= i < N and 0 <= j < M and 0 <= k < K }"
                        , reductionRead = "[N,M,K] -> { S1[i,j,k] -> C[i,j] }"
                        , reductionWrite = "[N,M,K] -> { S1[i,j,k] -> C[i,j] }"
                        }

            lowerProgram front `shouldBe` Right expected

        it "lowers 4D mixed reduction and non-reduction statements with stable stage order" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "b", extent = "Batch"}
                                , Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                , Axis{iter = "k", extent = "K"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["Batch", "N", "K"]}
                                , TensorDecl{tensor = "W", shape = ["Batch", "K", "M"]}
                                , TensorDecl{tensor = "Bias", shape = ["Batch", "N", "M"]}
                                , TensorDecl{tensor = "Acc", shape = ["Batch", "N", "M"]}
                                , TensorDecl{tensor = "Y", shape = ["Batch", "N", "M"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "Acc"
                                    , outputIndex = [IxVar "b", IxVar "i", IxVar "j"]
                                    , rhs = EConst 0
                                    }
                                , Reduction
                                    { reductionOp = ReduceAdd
                                    , outputTensor = "Acc"
                                    , outputIndex = [IxVar "b", IxVar "i", IxVar "j"]
                                    , rhs =
                                        EMul
                                            (ELoad "A" [IxVar "b", IxVar "i", IxVar "k"])
                                            (ELoad "W" [IxVar "b", IxVar "k", IxVar "j"])
                                    }
                                , Assign
                                    { outputTensor = "Y"
                                    , outputIndex = [IxVar "b", IxVar "i", IxVar "j"]
                                    , rhs =
                                        EAdd
                                            (ELoad "Acc" [IxVar "b", IxVar "i", IxVar "j"])
                                            (ELoad "Bias" [IxVar "b", IxVar "i", IxVar "j"])
                                    }
                                ]
                        }

                expected =
                    RawPolyhedralModel
                        { context = "[Batch,N,M,K] -> { : 0 <= Batch and 0 <= N and 0 <= M and 0 <= K }"
                        , domain =
                            "[Batch,N,M,K] -> { S0[b,i,j] : 0 <= b < Batch and 0 <= i < N and 0 <= j < M; S1[b,i,j,k] : 0 <= b < Batch and 0 <= i < N and 0 <= j < M and 0 <= k < K; S2[b,i,j] : 0 <= b < Batch and 0 <= i < N and 0 <= j < M }"
                        , programOrder =
                            "[Batch,N,M,K] -> { S0[b,i,j] -> [0,b,i,j,0]; S1[b,i,j,k] -> [1,b,i,j,k]; S2[b,i,j] -> [2,b,i,j,0] }"
                        , readAccess =
                            "[Batch,N,M,K] -> { S1[b,i,j,k] -> A[b,i,k]; S1[b,i,j,k] -> W[b,k,j]; S2[b,i,j] -> Acc[b,i,j]; S2[b,i,j] -> Bias[b,i,j] }"
                        , writeAccess =
                            "[Batch,N,M,K] -> { S0[b,i,j] -> Acc[b,i,j]; S1[b,i,j,k] -> Acc[b,i,j]; S2[b,i,j] -> Y[b,i,j] }"
                        , reductionDomain =
                            "[Batch,N,M,K] -> { S1[b,i,j,k] : 0 <= b < Batch and 0 <= i < N and 0 <= j < M and 0 <= k < K }"
                        , reductionRead = "[Batch,N,M,K] -> { S1[b,i,j,k] -> Acc[b,i,j] }"
                        , reductionWrite = "[Batch,N,M,K] -> { S1[b,i,j,k] -> Acc[b,i,j] }"
                        }

            lowerProgram front `shouldBe` Right expected

        it "zero pads omitted axes in multi-statement stage-first order tuples" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                , Axis{iter = "k", extent = "K"}
                                , Axis{iter = "l", extent = "L"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                                , TensorDecl{tensor = "B", shape = ["N", "M"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                , TensorDecl{tensor = "D", shape = ["N", "M"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs =
                                        EAdd
                                            (ELoad "A" [IxVar "i", IxVar "j"])
                                            (ELoad "B" [IxVar "i", IxVar "j"])
                                    }
                                , Assign
                                    { outputTensor = "D"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = ELoad "C" [IxVar "i", IxVar "j"]
                                    }
                                ]
                        }

                expected =
                    RawPolyhedralModel
                        { context = "[N,M,K,L] -> { : 0 <= N and 0 <= M and 0 <= K and 0 <= L }"
                        , domain =
                            "[N,M,K,L] -> { S0[i,j] : 0 <= i < N and 0 <= j < M; S1[i,j] : 0 <= i < N and 0 <= j < M }"
                        , programOrder =
                            "[N,M,K,L] -> { S0[i,j] -> [0,i,j,0,0]; S1[i,j] -> [1,i,j,0,0] }"
                        , readAccess = "[N,M,K,L] -> { S0[i,j] -> A[i,j]; S0[i,j] -> B[i,j]; S1[i,j] -> C[i,j] }"
                        , writeAccess = "[N,M,K,L] -> { S0[i,j] -> C[i,j]; S1[i,j] -> D[i,j] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

            lowerProgram front `shouldBe` Right expected

        it "fails multi-statement lowering when reduction would need multiple reduced axes" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [ Axis{iter = "i", extent = "N"}
                                , Axis{iter = "j", extent = "M"}
                                , Axis{iter = "k", extent = "K"}
                                , Axis{iter = "l", extent = "L"}
                                ]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N", "M", "K", "L"]}
                                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = EConst 0
                                    }
                                , Reduction
                                    { reductionOp = ReduceAdd
                                    , outputTensor = "C"
                                    , outputIndex = [IxVar "i", IxVar "j"]
                                    , rhs = ELoad "A" [IxVar "i", IxVar "j", IxVar "k", IxVar "l"]
                                    }
                                ]
                        }

            lowerProgram invalid
                `shouldBe` Left (ErrMultiStmtRequiresSingleReductionAxis ["k", "l"])
