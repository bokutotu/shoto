module ShotoSpec (spec) where

import           ISL        (AstExpression (..), AstOp (..), AstTree (..))
import           Shoto      (compile)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Shoto compile" $ do
        it "can compute RAW dependence for simple copy" $ do
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
