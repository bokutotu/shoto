module ShotoSpec (spec) where

import           ISL        (AstExpression (..), AstOp (..), AstTree (..))
import           Shoto      (compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "simple ast" $ do
            let domain = "[N] -> { S[i] : 0 <= i < N }"
                write = "{ S[i] -> A[i] }"
                reed = "{ S[i] -> B[i] }"
                schedule = "{ S[i] -> [i] }"
                params = ["N"]

                opCall = OpCall (ExprId "S") [ExprId "c0"]
                userAst = AstUser $ ExprOp opCall
                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = userAst
                        }

            result <- compile domain write reed schedule params
            result `shouldBe` Right expectedAst

        it "swap axis" $ do
            let domain = "[N,M] -> { S[i,j] : 0 <= i < N and 0 <= j < M }"
                write = "{ S[i,j] -> A[i][j] }"
                reed = "{ S[i,j] -> B[i][j] }"
                schedule = "{ S[i,j] -> [j,i] }"
                params = ["N", "M"]

                opCall = OpCall (ExprId "S") [ExprId "c1", ExprId "c0"]
                userAst = AstUser $ ExprOp opCall
                innerFor =
                    AstFor
                        { forIterator = "c1"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = userAst
                        }
                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "M"))
                        , forInc = ExprInt 1
                        , forBody = innerFor
                        }

            result <- compile domain write reed schedule params
            result `shouldBe` Right expectedAst
