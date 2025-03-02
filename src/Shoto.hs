module Shoto (someFunc, add) where

someFunc :: IO ()
someFunc = putStrLn "someFunc"

add :: Int -> Int -> Int
add x y = x + y
