{-# LANGUAGE DuplicateRecordFields #-}

module Tensor (Shape (..), Tensor (..)) where

newtype Shape = Shape {shape :: [Int]}

data Tensor = Tensor {shape :: Shape, stride :: Shape}
