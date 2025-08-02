import Runtime
import Foreign.C

main :: IO ()
main = do
    putStrLn "Testing CUDA allocation..."
    let testData = [1.0, 2.0, 3.0, 4.0] :: [CFloat]
    putStrLn $ "Input data: " ++ show testData
    
    -- Test GPU allocation
    gpu <- copyToGpu testData
    putStrLn "GPU allocation successful!"
    
    -- Copy back
    result <- copyToCpu gpu
    putStrLn $ "Result: " ++ show result