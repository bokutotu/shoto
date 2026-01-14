module ISL.UnionSetSpec (spec) where

import           Data.List  (sortOn)
import           ISL
import qualified ISL.Ast    as Ast
import           Test.Hspec

normalizeUnionSet :: String -> Either String Ast.UnionSetExpr
normalizeUnionSet str = do
    Ast.UnionSetExpr sets <- Ast.parseUnionSetExpr str
    pure $ Ast.UnionSetExpr (sortOn Ast.setExprToString sets)

normalizeResult :: Either IslError String -> Either String Ast.UnionSetExpr
normalizeResult = either (Left . show) normalizeUnionSet

spec :: Spec
spec = do
    describe "ISL Union Set" $ do
        it "can parse and convert union set to string" $ do
            result <- runISL $ do
                uset <- unionSet "{ A[i] : 0 <= i < 10; B[j] : 0 <= j < 5 }"
                unionSetToString uset
            normalizeResult result
                `shouldBe` normalizeUnionSet "{ A[i] : 0 <= i <= 9; B[j] : 0 <= j <= 4 }"

        it "can compute union set union" $ do
            result <- runISL $ do
                u1 <- unionSet "{ A[i] : 0 <= i < 5 }"
                u2 <- unionSet "{ B[j] : 0 <= j < 3 }"
                combined <- unionSetUnion u1 u2
                unionSetToString combined
            normalizeResult result
                `shouldBe` normalizeUnionSet "{ A[i] : 0 <= i <= 4; B[j] : 0 <= j <= 2 }"
