{-# LANGUAGE OverloadedStrings #-}

module ShotoSpec (spec) where

import qualified Data.List.NonEmpty as NE
import           FrontendIR         (Axis (..), Expr (..), FrontendError (..),
                                     IxExpr (..), Program (..), Stmt (..),
                                     TensorDecl (..))
import           Polyhedral         (ScheduleOptimization (..))
import           Polyhedral.Error   (IslError (..), OptimizeError (..),
                                     PolyhedralError (..))
import           Shoto              (CompileError (..), CudaDim (..),
                                     DeviceConfig (..), compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "compiles a simple 1D program to C code" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N"]}
                                , TensorDecl{tensor = "B", shape = ["N"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "A"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = ELoad "B" [IxVar "i"]
                                    }
                                ]
                        }

                expected =
                    unlines
                        [ "void shoto_kernel(int N, float* A, float* B) {"
                        , "    for (int c0 = 0; c0 < N; c0 += 1) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            result <- compile [] front CPU
            result `shouldBe` Right expected

        it "compiles a simple 1D program to CUDA code" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N"]}
                                , TensorDecl{tensor = "B", shape = ["N"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "A"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = ELoad "B" [IxVar "i"]
                                    }
                                ]
                        }

                expected =
                    unlines
                        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B) {"
                        , "    int c0 = blockIdx.x * blockDim.x + threadIdx.x;"
                        , "    if (c0 < N) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            result <- compile [] front (GPU CudaX)
            result `shouldBe` Right expected

        it "returns Frontend errors as CompileFrontendError" $ do
            let invalid =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N"]}
                                , TensorDecl{tensor = "B", shape = ["N"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "A"
                                    , outputIndex = [IxVar "j"]
                                    , rhs = ELoad "B" [IxVar "i"]
                                    }
                                ]
                        }

            result <- compile [] invalid CPU
            result `shouldBe` Left (CompileFrontendError (ErrStoreIndexMismatch ["i"] ["j"]))

        it "returns Polyhedral errors as CompilePolyhedralError" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N"]}
                                , TensorDecl{tensor = "B", shape = ["N"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "A"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = ELoad "B" [IxVar "i"]
                                    }
                                ]
                        }

            result <- compile [Tile []] front CPU

            result
                `shouldBe` Left
                    (CompilePolyhedralError (PolyhedralOptimizeError OptimizeTileNoAxis Nothing))

        it "keeps internal ISL payload inside typed optimize errors" $ do
            let front =
                    Program
                        { axes =
                            NE.fromList
                                [Axis{iter = "i", extent = "N"}]
                        , tensors =
                            NE.fromList
                                [ TensorDecl{tensor = "A", shape = ["N"]}
                                , TensorDecl{tensor = "B", shape = ["N"]}
                                ]
                        , stmts =
                            NE.fromList
                                [ Assign
                                    { outputTensor = "A"
                                    , outputIndex = [IxVar "i"]
                                    , rhs = ELoad "B" [IxVar "i"]
                                    }
                                ]
                        }

            result <- compile [LoopInterchange [1, 1]] front CPU

            let payloadIsPresent =
                    case result of
                        Left
                            ( CompilePolyhedralError
                                    (PolyhedralOptimizeError OptimizeInternalFailure (Just IslError{islFunction = fn}))
                                ) -> not (null fn)
                        _ -> False

            payloadIsPresent `shouldBe` True
