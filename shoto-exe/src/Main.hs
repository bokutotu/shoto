module Main (main) where

import           Shoto.Backend
import           Shoto.Codegen
import           Shoto.Frontend
import           Shoto.Middleend

main :: IO ()
main = putStrLn $ "Shoto Polyhedral Compiler: " ++ codegenFunc
