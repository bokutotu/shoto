module TinyIRSpec where

import qualified Data.Map              as M
import           Foreign.C             (CFloat)
import qualified IR
import           Runtime
import           Test.Hspec
import qualified TinyIR

spec :: Spec
spec = describe "TinyIR" $ do
    describe "Binary operations via IR" $ do
        it "executes Add via IR" $
            testBinaryOpIR TinyIR.Add [1.0, 2.0, 3.0] [4.0, 5.0, 6.0] [5.0, 7.0, 9.0]
        it "executes Sub via IR" $
            testBinaryOpIR TinyIR.Sub [5.0, 7.0, 9.0] [2.0, 3.0, 4.0] [3.0, 4.0, 5.0]
        it "executes Mul via IR" $
            testBinaryOpIR TinyIR.Mul [2.0, 3.0, 4.0] [5.0, 6.0, 7.0] [10.0, 18.0, 28.0]
        it "executes Div via IR" $
            testBinaryOpIR TinyIR.Div [10.0, 15.0, 20.0] [2.0, 3.0, 4.0] [5.0, 5.0, 5.0]
    
    describe "Reduce operations via IR" $ do
        it "reduces all elements to scalar (Sum) via IR" $
            testReduceAllIR TinyIR.Sum [1.0, 2.0, 3.0, 4.0, 5.0] 15.0
        it "reduces all elements to scalar (Max) via IR" $
            testReduceAllIR TinyIR.Max [1.0, 5.0, 3.0, 2.0, 4.0] 5.0
        it "reduces all elements to scalar (Min) via IR" $
            testReduceAllIR TinyIR.Min [5.0, 2.0, 8.0, 1.0, 3.0] 1.0
        
        it "reduces along axis (Sum) via IR" $
            -- Shape [2, 3] -> axis=1 -> [2]
            -- [[1,2,3], [4,5,6]] -> [6, 15]
            testReduceAxisIR TinyIR.Sum 1 [TinyIR.Static 2, TinyIR.Static 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [6.0, 15.0]

-- Test binary operations using IR
testBinaryOpIR :: TinyIR.BinaryTy -> [CFloat] -> [CFloat] -> [CFloat] -> IO ()
testBinaryOpIR ty a b expected = do
    let size = length a
        -- Build IR graph
        input1 = IR.Input (TinyIR.Input [TinyIR.Static size])
        input2 = IR.Input (TinyIR.Input [TinyIR.Static size])
        binOp = IR.Operation (TinyIR.ElementWise (TinyIR.Binary ty)) [0, 1]
        ir = M.fromList [(0, input1), (1, input2), (2, binOp)] :: TinyIR.TinyIR
        outputInit = replicate size 0.0 :: [CFloat]
    
    -- Prepare GPU buffers
    aGpu <- copyToGpu a
    bGpu <- copyToGpu b
    cGpu <- copyToGpu outputInit
    
    -- Compile and execute via IR
    compiledGraph <- compileGraph ir [2]
    executeGraph ir compiledGraph [(0, aGpu), (1, bGpu)] (2, cGpu)
    
    -- Check result
    result <- copyToCpu cGpu
    result `shouldBe` expected

-- Test reduce all operations using IR
testReduceAllIR :: TinyIR.ReduceOp -> [CFloat] -> CFloat -> IO ()
testReduceAllIR op input expected = do
    let size = length input
        -- Build IR graph for full reduction
        input1 = IR.Input (TinyIR.Input [TinyIR.Static size])
        reduceOp = IR.Operation (TinyIR.Reduce op Nothing) [0]
        ir = M.fromList [(0, input1), (1, reduceOp)] :: TinyIR.TinyIR
    
    -- Prepare GPU buffers
    inputGpu <- copyToGpu input
    outputGpu <- copyToGpu [0.0 :: CFloat]  -- Single element for scalar output
    
    -- Compile and execute via IR
    compiledGraph <- compileGraph ir [1]
    executeGraph ir compiledGraph [(0, inputGpu)] (1, outputGpu)
    
    -- Check result
    [result] <- copyToCpu outputGpu
    result `shouldBe` expected

-- Test reduce along axis operations using IR
testReduceAxisIR :: TinyIR.ReduceOp -> Int -> TinyIR.Shape -> [CFloat] -> [CFloat] -> IO ()
testReduceAxisIR op axis shape input expected = do
    let totalSize = product [n | TinyIR.Static n <- shape]
        outputSize = product [n | (i, TinyIR.Static n) <- zip [0..] shape, i /= axis]
        -- Build IR graph for axis reduction
        input1 = IR.Input (TinyIR.Input shape)
        reduceOp = IR.Operation (TinyIR.Reduce op (Just axis)) [0]
        ir = M.fromList [(0, input1), (1, reduceOp)] :: TinyIR.TinyIR
    
    -- Prepare GPU buffers
    inputGpu <- copyToGpu input
    outputGpu <- copyToGpu (replicate outputSize 0.0)
    
    -- Compile and execute via IR
    compiledGraph <- compileGraph ir [1]
    executeGraph ir compiledGraph [(0, inputGpu)] (1, outputGpu)
    
    -- Check result
    result <- copyToCpu outputGpu
    result `shouldBe` expected

