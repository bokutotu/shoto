{-# LANGUAGE OverloadedStrings #-}

module ShotoSpec (spec) where

import qualified Data.List.NonEmpty as NE
import           FrontendIR         (Axis (..), Expr (..), FrontendError (..),
                                     IxExpr (..), Program (..), Stmt (..),
                                     TensorDecl (..))
import           Polyhedral         (ScheduleOptimization (..))
import           Polyhedral.Error   (IslError (..), OptimizeError (..),
                                     PolyhedralError (..))
import           Shoto              (CompileError (..), DeviceConfig (..),
                                     compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "compiles a simple 1D program to C code" $ do
            let expected =
                    unlines
                        [ "void shoto_kernel(int N, float* A, float* B) {"
                        , "    for (int c0 = 0; c0 < N; c0 += 1) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            result <- compile [] simpleCopyProgram CPU
            result `shouldBe` Right expected

        it "compiles a simple 1D program to CUDA code" $ do
            let expected =
                    unlines
                        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B) {"
                        , "    int c0 = blockIdx.x * blockDim.x + threadIdx.x;"
                        , "    if (c0 < N) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            result <- compile [] simpleCopyProgram GPU
            result `shouldBe` Right expected

        it "compiles a simple 2D program to C code" $ do
            let expected =
                    unlines
                        [ "void shoto_kernel(int N, int M, float* A, float* B) {"
                        , "    for (int c0 = 0; c0 < N; c0 += 1) {"
                        , "        for (int c1 = 0; c1 < M; c1 += 1) {"
                        , "            A[((c0 * M) + c1)] = B[((c0 * M) + c1)];"
                        , "        }"
                        , "    }"
                        , "}"
                        ]

            result <- compile [] simpleCopy2DProgram CPU
            result `shouldBe` Right expected

        it "compiles a simple 2D program to CUDA code" $ do
            let expected =
                    unlines
                        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, int M, float* A, float* B) {"
                        , "    int c1 = blockIdx.x * blockDim.x + threadIdx.x;"
                        , "    int c0 = blockIdx.y * blockDim.y + threadIdx.y;"
                        , "    if ((c0 < N) && (c1 < M)) {"
                        , "        A[((c0 * M) + c1)] = B[((c0 * M) + c1)];"
                        , "    }"
                        , "}"
                        ]

            result <- compile [] simpleCopy2DProgram GPU
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
            result <- compile [Tile []] simpleCopyProgram CPU

            result
                `shouldBe` Left
                    (CompilePolyhedralError (PolyhedralOptimizeError OptimizeTileNoAxis Nothing))

        it "keeps internal ISL payload inside typed optimize errors" $ do
            result <- compile [LoopInterchange [1, 1]] simpleCopyProgram CPU

            let payloadIsPresent =
                    case result of
                        Left
                            ( CompilePolyhedralError
                                    (PolyhedralOptimizeError OptimizeInternalFailure (Just IslError{islFunction = fn}))
                                ) -> not (null fn)
                        _ -> False

            payloadIsPresent `shouldBe` True

simpleCopyProgram :: Program
simpleCopyProgram =
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

simpleCopy2DProgram :: Program
simpleCopy2DProgram =
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
                ]
        , stmts =
            NE.fromList
                [ Assign
                    { outputTensor = "A"
                    , outputIndex = [IxVar "i", IxVar "j"]
                    , rhs = ELoad "B" [IxVar "i", IxVar "j"]
                    }
                ]
        }
