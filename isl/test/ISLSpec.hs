module ISLSpec (spec) where

import Test.Hspec

spec :: Spec
spec = do
  describe "ISL" $ do
    it "placeholder test" $ do
      True `shouldBe` True
