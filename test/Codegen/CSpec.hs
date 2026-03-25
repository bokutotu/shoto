{-# LANGUAGE OverloadedStrings #-}

module Codegen.CSpec (spec) where

import           Codegen             (generateC)
import qualified Data.List.NonEmpty  as NE
import           FrontendIR          (Axis (..), Expr (..), IxExpr (..),
                                      Program (..), Stmt (..), TensorDecl (..))
import           IR.Name             (KernelName (..), TensorName (..))
import           Polyhedral.Internal (AstExpression (..), AstOp (..),
                                      AstTree (..))
import           Test.Hspec

spec :: Spec
spec = do
    describe "Codegen.C" $ do
        it "generates C code for one statement and one index" $ do
            let expected =
                    unlines
                        [ "void shoto_kernel(int N, float* A, float* B) {"
                        , "    for (int c0 = 0; c0 < N; c0 += 1) {"
                        , "        A[c0] = B[c0];"
                        , "    }"
                        , "}"
                        ]

            generateC simpleCopyAst simpleCopyProgram `shouldBe` Right expected

        it "generates C code for a 2D statement with row-major flattening" $ do
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

            generateC simpleCopy2DAst simpleCopy2DProgram `shouldBe` Right expected

        it "preserves AST loop order for interchanged nests" $ do
            let expected =
                    unlines
                        [ "void shoto_kernel(int N, int M, float* A, float* B) {"
                        , "    for (int c1 = 0; c1 < M; c1 += 1) {"
                        , "        for (int c0 = 0; c0 < N; c0 += 1) {"
                        , "            A[((c0 * M) + c1)] = B[((c0 * M) + c1)];"
                        , "        }"
                        , "    }"
                        , "}"
                        ]

            generateC interchanged2DAst simpleCopy2DProgram `shouldBe` Right expected

        it "constructs shared IR names from string literals" $ do
            ("shoto_kernel" :: KernelName) `shouldBe` KernelName "shoto_kernel"
            ("A" :: TensorName) `shouldBe` TensorName "A"

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

interchanged2DAst :: AstTree
interchanged2DAst =
    AstFor
        { forIterator = "c1"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "M"))
        , forInc = ExprInt 1
        , forBody =
            AstFor
                { forIterator = "c0"
                , forInit = ExprInt 0
                , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                , forInc = ExprInt 1
                , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0", ExprId "c1"]
                }
        }

simpleCopyAst :: AstTree
simpleCopyAst =
    AstFor
        { forIterator = "c0"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
        , forInc = ExprInt 1
        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0"]
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
