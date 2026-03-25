{-# LANGUAGE OverloadedStrings #-}

module Codegen.CUDASpec (spec) where

import           Codegen.CUDA.Ast    (CudaAstError (..), lowerToCudaProgram)
import           Codegen.CUDA.Emit   (emitCudaProgram)
import           Codegen.GenIR       (buildGenProgram)
import           Data.Bifunctor      (first)
import qualified Data.List.NonEmpty  as NE
import           FrontendIR          (Axis (..), Expr (..), IxExpr (..),
                                      Program (..), Stmt (..), TensorDecl (..))
import           IR.Name             (KernelName (..), TensorName (..))
import           Polyhedral.Internal (AstExpression (..), AstOp (..),
                                      AstTree (..))
import           Test.Hspec

spec :: Spec
spec = do
    describe "Codegen.CUDA" $ do
        it "generates CUDA code for a 1D statement" $ do
            let expected =
                    unlines
                        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B) {"
                        , "    int c0 = blockIdx.x * blockDim.x + threadIdx.x;"
                        , "    if (c0 < N) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            cudaSource simpleCopyAst simpleCopyProgram
                `shouldBe` Right expected

        it "generates CUDA code for a 2D statement with logical axis mapping" $ do
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

            cudaSource simpleCopy2DAst simpleCopy2DProgram
                `shouldBe` Right expected

        it "generates CUDA code for a 3D statement with z/y/x mapping" $ do
            let expected =
                    unlines
                        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, int M, int K, float* A, float* B) {"
                        , "    int c2 = blockIdx.x * blockDim.x + threadIdx.x;"
                        , "    int c1 = blockIdx.y * blockDim.y + threadIdx.y;"
                        , "    int c0 = blockIdx.z * blockDim.z + threadIdx.z;"
                        , "    if ((c0 < N) && (c1 < M) && (c2 < K)) {"
                        , "        A[((((c0 * M) + c1) * K) + c2)] = B[((((c0 * M) + c1) * K) + c2)];"
                        , "    }"
                        , "}"
                        ]

            cudaSource simpleCopy3DAst simpleCopy3DProgram
                `shouldBe` Right expected

        it "rejects rank greater than 3" $ do
            cudaSource simpleCopy4DAst simpleCopy4DProgram
                `shouldBe` Left (show (ErrCudaAstRankTooLarge 4))

        it "constructs shared IR names from string literals" $ do
            ("shoto_kernel_cuda" :: KernelName)
                `shouldBe` KernelName "shoto_kernel_cuda"
            ("B" :: TensorName) `shouldBe` TensorName "B"

simpleCopyAst :: AstTree
simpleCopyAst =
    AstFor
        { forIterator = "c0"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
        , forInc = ExprInt 1
        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0"]
        }

simpleCopy2DAst :: AstTree
simpleCopy2DAst =
    AstFor
        { forIterator = "c0"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
        , forInc = ExprInt 1
        , forBody =
            AstFor
                { forIterator = "c1"
                , forInit = ExprInt 0
                , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "M"))
                , forInc = ExprInt 1
                , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0", ExprId "c1"]
                }
        }

simpleCopy3DAst :: AstTree
simpleCopy3DAst =
    AstFor
        { forIterator = "c0"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
        , forInc = ExprInt 1
        , forBody =
            AstFor
                { forIterator = "c1"
                , forInit = ExprInt 0
                , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "M"))
                , forInc = ExprInt 1
                , forBody =
                    AstFor
                        { forIterator = "c2"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c2") (ExprId "K"))
                        , forInc = ExprInt 1
                        , forBody =
                            AstUser $
                                ExprOp $
                                    OpCall (ExprId "S0") [ExprId "c0", ExprId "c1", ExprId "c2"]
                        }
                }
        }

simpleCopy4DAst :: AstTree
simpleCopy4DAst =
    AstFor
        { forIterator = "c0"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
        , forInc = ExprInt 1
        , forBody =
            AstFor
                { forIterator = "c1"
                , forInit = ExprInt 0
                , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "M"))
                , forInc = ExprInt 1
                , forBody =
                    AstFor
                        { forIterator = "c2"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c2") (ExprId "K"))
                        , forInc = ExprInt 1
                        , forBody =
                            AstFor
                                { forIterator = "c3"
                                , forInit = ExprInt 0
                                , forCond = ExprOp (OpLt (ExprId "c3") (ExprId "L"))
                                , forInc = ExprInt 1
                                , forBody =
                                    AstUser $
                                        ExprOp $
                                            OpCall
                                                (ExprId "S0")
                                                [ ExprId "c0"
                                                , ExprId "c1"
                                                , ExprId "c2"
                                                , ExprId "c3"
                                                ]
                                }
                        }
                }
        }

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

simpleCopy3DProgram :: Program
simpleCopy3DProgram =
    Program
        { axes =
            NE.fromList
                [ Axis{iter = "i", extent = "N"}
                , Axis{iter = "j", extent = "M"}
                , Axis{iter = "k", extent = "K"}
                ]
        , tensors =
            NE.fromList
                [ TensorDecl{tensor = "A", shape = ["N", "M", "K"]}
                , TensorDecl{tensor = "B", shape = ["N", "M", "K"]}
                ]
        , stmts =
            NE.fromList
                [ Assign
                    { outputTensor = "A"
                    , outputIndex = [IxVar "i", IxVar "j", IxVar "k"]
                    , rhs = ELoad "B" [IxVar "i", IxVar "j", IxVar "k"]
                    }
                ]
        }

simpleCopy4DProgram :: Program
simpleCopy4DProgram =
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
                , TensorDecl{tensor = "B", shape = ["N", "M", "K", "L"]}
                ]
        , stmts =
            NE.fromList
                [ Assign
                    { outputTensor = "A"
                    , outputIndex = [IxVar "i", IxVar "j", IxVar "k", IxVar "l"]
                    , rhs = ELoad "B" [IxVar "i", IxVar "j", IxVar "k", IxVar "l"]
                    }
                ]
        }

cudaSource :: AstTree -> Program -> Either String String
cudaSource ast program = do
    genProgram <- first show $ buildGenProgram ast program
    cudaProgram <- first show $ lowerToCudaProgram genProgram
    pure $ emitCudaProgram cudaProgram
