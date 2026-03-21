{-# LANGUAGE OverloadedStrings #-}

module CodegenSpec (spec) where

import           Codegen             (CodegenError (..), CudaDim (..),
                                      generateC, generateCuda)
import           Codegen.C.Ast       (CFunctionName (..), CTensorName (..))
import           Codegen.CUDA.Ast    (CudaKernelName (..), CudaTensorName (..))
import           Codegen.GenIR       (GenIRError (..))
import qualified Data.List.NonEmpty  as NE
import           FrontendIR          (Axis (..), Expr (..), IxExpr (..),
                                      Program (..), Stmt (..), TensorDecl (..))
import           Polyhedral.Internal (AstExpression (..), AstOp (..),
                                      AstTree (..))
import           Test.Hspec

spec :: Spec
spec = do
    describe "Codegen" $ do
        it "generates C code for one statement and one index" $ do
            let program = simpleCopyProgram
                expected =
                    unlines
                        [ "void shoto_kernel(int N, float* A, float* B) {"
                        , "    for (int c0 = 0; c0 < N; c0 += 1) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            let ast = simpleCopyAst
            generateC ast program `shouldBe` Right expected

        it "generates CUDA code with loop mapped to x" $ do
            let program = simpleCopyProgram
                expected =
                    unlines
                        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B) {"
                        , "    int c0 = blockIdx.x * blockDim.x + threadIdx.x;"
                        , "    if (c0 < N) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            let ast = simpleCopyAst
            generateCuda CudaX ast program `shouldBe` Right expected

        it "generates CUDA code with loop mapped to y" $ do
            let program = simpleCopyProgram
                expected =
                    unlines
                        [ "extern \"C\" __global__ void shoto_kernel_cuda(int N, float* A, float* B) {"
                        , "    int c0 = blockIdx.y * blockDim.y + threadIdx.y;"
                        , "    if (c0 < N) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            let ast = simpleCopyAst
            generateCuda CudaY ast program `shouldBe` Right expected

        it "constructs generated name wrappers from string literals" $ do
            ("shoto_kernel" :: CFunctionName) `shouldBe` CFunctionName "shoto_kernel"
            ("A" :: CTensorName) `shouldBe` CTensorName "A"
            ("shoto_kernel_cuda" :: CudaKernelName)
                `shouldBe` CudaKernelName "shoto_kernel_cuda"
            ("B" :: CudaTensorName) `shouldBe` CudaTensorName "B"

        it "rejects multi-axis program in this first-stage codegen" $ do
            let invalidProgram =
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

            let ast = simpleCopyAst
            generateC ast invalidProgram
                `shouldBe` Left
                    (CodegenGenIRError (ErrGenExpectedSingleAxis 2))

simpleCopyAst :: AstTree
simpleCopyAst =
    AstFor
        { forIterator = "c0"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
        , forInc = ExprInt 1
        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0"]
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
