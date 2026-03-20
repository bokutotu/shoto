{-# LANGUAGE OverloadedStrings #-}

module ShotoSpec (spec) where

import qualified Data.List.NonEmpty  as NE
import           FrontendIR          (Axis (..), Expr (..), FrontendError (..),
                                      IxExpr (..), Program (..), Stmt (..),
                                      TensorDecl (..))
import           Polyhedral          (ScheduleOptimization (..))
import           Polyhedral.Error    (IslError (..), OptimizeError (..),
                                      PolyhedralError (..))
import           Polyhedral.Internal (AstExpression (..), AstOp (..),
                                      AstTree (..))
import           Shoto               (CompileError (..), compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "compiles a simple 1D program to AST" $ do
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

                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0"]
                        }

            result <- compile [] front
            result `shouldBe` Right expectedAst

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

            result <- compile [] invalid
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

            result <- compile [Tile []] front

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

            result <- compile [LoopInterchange [1, 1]] front

            let payloadIsPresent =
                    case result of
                        Left
                            ( CompilePolyhedralError
                                    (PolyhedralOptimizeError OptimizeInternalFailure (Just IslError{islFunction = fn}))
                                ) -> not (null fn)
                        _ -> False

            payloadIsPresent `shouldBe` True
