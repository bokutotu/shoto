{-# LANGUAGE OverloadedStrings #-}

module ShotoSpec (spec) where

import           FrontendIR (Axis (..), Expr (..), FrontendError (..),
                             IxExpr (..), Program (..), Stmt (..))
import           ISL        (AstExpression (..), AstOp (..), AstTree (..),
                             IslError (..))
import           Polyhedral (ScheduleOptimization (..))
import           Shoto      (CompileError (..), compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "compiles a simple 1D program to AST" $ do
            let front =
                    Program
                        { axes =
                            [Axis{iter = "i", extent = "N"}]
                        , stmt =
                            Stmt
                                { outputTensor = "A"
                                , outputIndex = [IxVar "i"]
                                , rhs = ELoad "B" [IxVar "i"]
                                }
                        }

                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S") [ExprId "c0"]
                        }

            result <- compile [] front
            result `shouldBe` Right expectedAst

        it "returns Frontend errors as CompileFrontendError" $ do
            let invalid =
                    Program
                        { axes =
                            [Axis{iter = "i", extent = "N"}]
                        , stmt =
                            Stmt
                                { outputTensor = "A"
                                , outputIndex = [IxVar "j"]
                                , rhs = ELoad "B" [IxVar "i"]
                                }
                        }

            result <- compile [] invalid
            result `shouldBe` Left (CompileFrontendError (ErrStoreIndexMismatch ["i"] ["j"]))

        it "returns ISL errors as CompileIslError" $ do
            let front =
                    Program
                        { axes =
                            [Axis{iter = "i", extent = "N"}]
                        , stmt =
                            Stmt
                                { outputTensor = "A"
                                , outputIndex = [IxVar "i"]
                                , rhs = ELoad "B" [IxVar "i"]
                                }
                        }

            result <- compile [Tile []] front

            let actualFunction =
                    case result of
                        Left (CompileIslError islErr) -> Just (islFunction islErr)
                        _ -> Nothing

            actualFunction `shouldBe` Just "applyScheduleOptimization(Tile): at least one axis is required"

        it "keeps ISL error payload" $ do
            let front =
                    Program
                        { axes =
                            [Axis{iter = "i", extent = "N"}]
                        , stmt =
                            Stmt
                                { outputTensor = "A"
                                , outputIndex = [IxVar "i"]
                                , rhs = ELoad "B" [IxVar "i"]
                                }
                        }

            result <- compile [Tile []] front

            let payloadIsPresent =
                    case result of
                        Left (CompileIslError IslError{islFunction = fn}) -> not (null fn)
                        _ -> False

            payloadIsPresent `shouldBe` True
