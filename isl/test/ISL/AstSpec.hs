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
                }
        refN = Ast.DimRef Ast.ParamDim paramN
        refI = Ast.DimRef Ast.InDim dimI
        linearConst n = Ast.LinearExpr space n Map.empty
        linearVar ref = Ast.LinearExpr space 0 (Map.singleton ref 1)
        affineConst n = Ast.AffineLinear (linearConst n)
        affineVar ref = Ast.AffineLinear (linearVar ref)
        lower = Ast.Constraint Ast.RelLe (affineConst 0) (affineVar refI)
        upper = Ast.Constraint Ast.RelLe (affineVar refI) (affineVar refN)
     in Ast.SetExpr space [lower, upper]
