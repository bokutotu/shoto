{-# LANGUAGE DuplicateRecordFields #-}

module Tensor (Shape (..), Tensor (..), shapeIdx) where

newtype Shape = Shape {shape :: [Int]}

shapeIdx :: Shape -> Int -> Int
shapeIdx Shape{shape} idx = shape !! idx

data Tensor = Tensor {shape :: Shape, stride :: Shape}
