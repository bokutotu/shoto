module Polyhedral.Internal.ContextSpec (spec) where

import           Polyhedral.Internal
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL Context" $ do
        it "can run ISL monad" $ do
            result <- runISL $ do
                s <- set "{ [i] : 0 <= i < 10 }"
                setToString s
            result `shouldBe` Right "{ [i] : 0 <= i <= 9 }"
