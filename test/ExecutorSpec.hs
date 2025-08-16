module ExecutorSpec where

import qualified Data.Map   as M
import           Executor   (compileGraph, executeGraph)
import           Foreign.C  (CFloat (..))
import qualified IR
import           Runtime    (copyToCpu, copyToGpu)
import           Test.Hspec (Spec, describe, it, shouldBe)
import qualified TinyIR     as TIR

spec :: Spec
spec = describe "executor test" $ do
    it "simple test" $ do
        let input1 = IR.Input (TIR.Input [TIR.Static 10])
            input2 = IR.Input (TIR.Input [TIR.Static 10])
            add = IR.Operation (TIR.ElementWise (TIR.Binary TIR.Add)) [0, 1]
            ir = M.fromList [(0, input1), (1, input2), (2, add)] :: TIR.TinyIR
            input1Arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] :: [CFloat]
            input2Arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] :: [CFloat]
            outputArr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] :: [CFloat]
        input1Gpu <- copyToGpu input1Arr
        input2Gpu <- copyToGpu input2Arr
        outputGpu <- copyToGpu outputArr
        functions <- compileGraph ir [2]
        _ <- executeGraph ir functions [(0, input1Gpu), (1, input2Gpu)] (2, outputGpu)
        result <- copyToCpu outputGpu
        result `shouldBe` [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
