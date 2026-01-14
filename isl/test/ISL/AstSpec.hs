{-# LANGUAGE OverloadedStrings #-}

module ISL.AstSpec (spec) where

import qualified Data.Map.Strict as Map
import           Data.Text       (Text)
import           ISL.Ast         (AffineExpr (..), Band (..), BasicMap (..),
                                  BasicSet (..), Constraint (..), DimKind (..),
                                  DimRef (..), DivDef (..), LinearExpr (..),
                                  LocalDim (..), LocalSpace (..), MapExpr (..),
                                  MultiAffineExpr (..), MultiPwAffineExpr (..),
                                  MultiUnionPwAffineExpr (..),
                                  PwAffineExpr (..), Relation (..),
                                  ScheduleTree (..), SetExpr (..), Space (..),
                                  SpaceDim (..), Tuple (..), UnionSetExpr (..))
import qualified ISL.Ast         as Ast
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Ast roundtrip" $ do
        it "roundtrips SetExpr" $ do
            roundtrip Ast.parseSetExpr Ast.setExprToString sampleSetExpr

        it "roundtrips MapExpr" $ do
            roundtrip Ast.parseMapExpr Ast.mapExprToString sampleMapExpr

        it "roundtrips PwAffineExpr" $ do
            roundtrip Ast.parsePwAffineExpr Ast.pwAffineExprToString samplePwAffineExpr

        it "roundtrips MultiAffineExpr" $ do
            roundtrip Ast.parseMultiAffineExpr Ast.multiAffineExprToString sampleMultiAffineExpr

        it "roundtrips MultiPwAffineExpr" $ do
            roundtrip Ast.parseMultiPwAffineExpr Ast.multiPwAffineExprToString sampleMultiPwAffineExpr

        it "roundtrips MultiUnionPwAffineExpr" $ do
            roundtrip
                Ast.parseMultiUnionPwAffineExpr
                Ast.multiUnionPwAffineExprToString
                sampleMultiUnionPwAffineExpr

        it "roundtrips ScheduleTree" $ do
            roundtrip Ast.parseScheduleTree Ast.scheduleTreeToString sampleScheduleTree

        it "normalizes strict inequalities to non-strict constraints" $ do
            let strict = "{ [i] : i < 10 }"
                nonStrict = "{ [i] : i <= 9 }"
            Ast.parseSetExpr strict `shouldBe` Ast.parseSetExpr nonStrict

        it "parses local dimensions in constraints" $ do
            let input = "{ [i] : exists (e0: i - 2*e0 = 0) }"
                space = mkSetSpace [] Nothing ["i"]
                locals = [LocalDim (Just "e0") Nothing]
                localSpace = LocalSpace space locals
                lhs = affine localSpace 0 [(DimRef InDim 0, 1), (DimRef LocalDimKind 0, -2)]
                rhs = affine localSpace 0 []
                constraint = Constraint RelEq lhs rhs
                expected = SetExpr space [BasicSet localSpace [constraint]]
            Ast.parseSetExpr input `shouldBe` Right expected

        it "parses floor divisions in constraints" $ do
            let input = "{ [i] : i - floor((i)/2) = 0 }"
                space = mkSetSpace [] Nothing ["i"]
                numerator = LinearExpr 0 (Map.singleton (DimRef InDim 0) 1)
                divDef = DivDef numerator 2
                localSpace = LocalSpace space [LocalDim Nothing (Just divDef)]
                lhs = affine localSpace 0 [(DimRef InDim 0, 1), (DimRef LocalDimKind 0, -1)]
                rhs = affine localSpace 0 []
                constraint = Constraint RelEq lhs rhs
                expected = SetExpr space [BasicSet localSpace [constraint]]
            Ast.parseSetExpr input `shouldBe` Right expected

roundtrip :: (Eq a, Show a) => (String -> Either String a) -> (a -> String) -> a -> Expectation
roundtrip parse render value =
    parse (render value) `shouldBe` Right value

sampleSetExpr :: SetExpr
sampleSetExpr =
    let space = mkSetSpace ["N"] (Just "S") ["i"]
        localSpace = LocalSpace space []
        lower = Constraint RelLe (affine localSpace 0 []) (affine localSpace 0 [(DimRef InDim 0, 1)])
        upper =
            Constraint
                RelLe
                (affine localSpace 0 [(DimRef InDim 0, 1)])
                (affine localSpace 0 [(DimRef ParamDim 0, 1)])
     in SetExpr space [BasicSet localSpace [lower, upper]]

sampleMapExpr :: MapExpr
sampleMapExpr =
    let space = mkMapSpace [] (Just "S") ["i"] (Just "T") ["j"]
        localSpace = LocalSpace space []
        lhs = affine localSpace 0 [(DimRef OutDim 0, 1)]
        rhs = affine localSpace 1 [(DimRef InDim 0, 1)]
        constraint = Constraint RelEq lhs rhs
     in MapExpr space [BasicMap localSpace [constraint]]

samplePwAffineExpr :: PwAffineExpr
samplePwAffineExpr =
    let space = mkSetSpace [] (Just "S") ["i"]
        localSpace = LocalSpace space []
        part = BasicSet localSpace []
        expr = affine localSpace 0 [(DimRef InDim 0, 1)]
     in PwAffineExpr space [(part, expr)]

sampleMultiAffineExpr :: MultiAffineExpr
sampleMultiAffineExpr =
    let inTuple = mkTuple (Just "S") ["i"]
        baseSpace = mkSetSpace [] (Just "S") ["i"]
        localSpace = LocalSpace baseSpace []
        expr0 = affine localSpace 0 [(DimRef InDim 0, 1)]
        expr1 = affine localSpace 1 [(DimRef InDim 0, 1)]
        multiSpace = mkMultiSpace [] inTuple 2
     in MultiAffineExpr multiSpace [expr0, expr1]

sampleMultiPwAffineExpr :: MultiPwAffineExpr
sampleMultiPwAffineExpr =
    let inTuple = mkTuple (Just "S") ["i"]
        baseSpace = mkSetSpace [] (Just "S") ["i"]
        localSpace = LocalSpace baseSpace []
        part = BasicSet localSpace []
        expr0 = affine localSpace 0 [(DimRef InDim 0, 1)]
        expr1 = affine localSpace 1 [(DimRef InDim 0, 1)]
        pw0 = PwAffineExpr baseSpace [(part, expr0)]
        pw1 = PwAffineExpr baseSpace [(part, expr1)]
        multiSpace = mkMultiSpace [] inTuple 2
     in MultiPwAffineExpr multiSpace [pw0, pw1]

sampleMultiUnionPwAffineExpr :: MultiUnionPwAffineExpr
sampleMultiUnionPwAffineExpr =
    MultiUnionPwAffineExpr
        [ identityMultiPw "S" "i"
        , identityMultiPw "T" "j"
        ]

sampleScheduleTree :: ScheduleTree
sampleScheduleTree =
    let domainSet = simpleDomainSet
        domain = UnionSetExpr [domainSet]
        schedule = MultiUnionPwAffineExpr [identityMultiPw "S" "i"]
        band =
            Band
                { bandSchedule = schedule
                , bandPermutable = True
                , bandCoincident = [True]
                , bandAstBuildOptions = Just domain
                }
     in TreeDomain domain [TreeBand band [TreeLeaf]]

simpleDomainSet :: SetExpr
simpleDomainSet =
    let space = mkSetSpace [] (Just "S") ["i"]
        localSpace = LocalSpace space []
     in SetExpr space [BasicSet localSpace []]

identityMultiPw :: Text -> Text -> MultiPwAffineExpr
identityMultiPw name dimName =
    let inTuple = mkTuple (Just name) [dimName]
        baseSpace = mkSetSpace [] (Just name) [dimName]
        localSpace = LocalSpace baseSpace []
        part = BasicSet localSpace []
        expr = affine localSpace 0 [(DimRef InDim 0, 1)]
        pw = PwAffineExpr baseSpace [(part, expr)]
        multiSpace = mkMultiSpace [] inTuple 1
     in MultiPwAffineExpr multiSpace [pw]

mkTuple :: Maybe Text -> [Text] -> Tuple
mkTuple name dims = Tuple name (map Ast.spaceDim dims)

mkSetSpace :: [Text] -> Maybe Text -> [Text] -> Space
mkSetSpace params name dims =
    Space
        { spaceParams = map Ast.spaceDim params
        , spaceIn = mkTuple name dims
        , spaceOut = Tuple Nothing []
        }

mkMapSpace :: [Text] -> Maybe Text -> [Text] -> Maybe Text -> [Text] -> Space
mkMapSpace params domName domDims ranName ranDims =
    Space
        { spaceParams = map Ast.spaceDim params
        , spaceIn = mkTuple domName domDims
        , spaceOut = mkTuple ranName ranDims
        }

mkMultiSpace :: [Text] -> Tuple -> Int -> Space
mkMultiSpace params inTuple outCount =
    Space
        { spaceParams = map Ast.spaceDim params
        , spaceIn = inTuple
        , spaceOut = Tuple Nothing (replicate outCount (SpaceDim Nothing))
        }

affine :: LocalSpace -> Integer -> [(DimRef, Integer)] -> AffineExpr
affine localSpace constant coeffs =
    AffineExpr localSpace (LinearExpr constant (Map.fromList coeffs))
