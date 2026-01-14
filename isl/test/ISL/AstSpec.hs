{-# LANGUAGE OverloadedStrings #-}

module ISL.AstSpec (spec) where

import qualified Data.Map.Strict as Map
import           ISL             (runISL, set, setToString)
import qualified ISL.Ast         as Ast
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Ast roundtrip" $ do
        it "roundtrips a set expression through ISL" $ do
            let setExpr = sampleSetExpr
            result <- runISL $ do
                s <- set (Ast.setExprToString setExpr)
                setToString s
            case result of
                Left err ->
                    expectationFailure ("ISL set roundtrip failed: " ++ show err)
                Right islStr ->
                    Ast.parseSetExpr islStr `shouldBe` Right setExpr

        it "normalizes strict inequalities to non-strict constraints" $ do
            let strict = "{ [i] : i < 10 }"
                nonStrict = "{ [i] : i <= 9 }"
            Ast.parseSetExpr strict `shouldBe` Ast.parseSetExpr nonStrict

        it "parses local dimensions in constraints" $ do
            let input = "{ [i] : exists (e0: i - 2*e0 = 0) }"
                dimI = Ast.spaceDim "i"
                dimE0 = Ast.spaceDim "e0"
                space =
                    Ast.Space
                        { Ast.spaceName = Nothing
                        , Ast.spaceParams = []
                        , Ast.spaceInputs = [dimI]
                        , Ast.spaceOutputs = []
                        , Ast.spaceLocals = [dimE0]
                        }
                refI = Ast.DimRef Ast.InDim dimI
                refE0 = Ast.DimRef Ast.LocalDim dimE0
                lhs = Ast.LinearExpr space 0 (Map.fromList [(refI, 1), (refE0, -2)]) []
                rhs = Ast.LinearExpr space 0 Map.empty []
                constraint = Ast.Constraint Ast.RelEq (Ast.AffineLinear lhs) (Ast.AffineLinear rhs)
                expected = Ast.SetExpr space [constraint]
            Ast.parseSetExpr input `shouldBe` Right expected

        it "parses floor divisions in constraints" $ do
            let input = "{ [i] : i - floor((i)/2) = 0 }"
                dimI = Ast.spaceDim "i"
                space =
                    Ast.Space
                        { Ast.spaceName = Nothing
                        , Ast.spaceParams = []
                        , Ast.spaceInputs = [dimI]
                        , Ast.spaceOutputs = []
                        , Ast.spaceLocals = []
                        }
                refI = Ast.DimRef Ast.InDim dimI
                numerator = Ast.LinearExpr space 0 (Map.singleton refI 1) []
                divExpr = Ast.DivExpr numerator 2
                lhs = Ast.LinearExpr space 0 (Map.singleton refI 1) [Ast.DivTerm (-1) divExpr]
                rhs = Ast.LinearExpr space 0 Map.empty []
                constraint = Ast.Constraint Ast.RelEq (Ast.AffineLinear lhs) (Ast.AffineLinear rhs)
                expected = Ast.SetExpr space [constraint]
            Ast.parseSetExpr input `shouldBe` Right expected

sampleSetExpr :: Ast.SetExpr
sampleSetExpr =
    let paramN = Ast.spaceDim "N"
        dimI = Ast.spaceDim "i"
        space =
            Ast.Space
                { Ast.spaceName = Just "S"
                , Ast.spaceParams = [paramN]
                , Ast.spaceInputs = [dimI]
                , Ast.spaceOutputs = []
                , Ast.spaceLocals = []
                }
        refN = Ast.DimRef Ast.ParamDim paramN
        refI = Ast.DimRef Ast.InDim dimI
        linearConst n = Ast.LinearExpr space n Map.empty []
        linearVar ref = Ast.LinearExpr space 0 (Map.singleton ref 1) []
        affineConst n = Ast.AffineLinear (linearConst n)
        affineVar ref = Ast.AffineLinear (linearVar ref)
        lower = Ast.Constraint Ast.RelLe (affineConst 0) (affineVar refI)
        upper = Ast.Constraint Ast.RelLe (affineVar refI) (affineVar refN)
     in Ast.SetExpr space [lower, upper]
