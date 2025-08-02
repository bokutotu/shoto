{-# LANGUAGE LambdaCase #-}

module FrontendIR (FrontendIR (..), opStr, gridDim) where

import           Tensor (Tensor (..), shapeIdx)

-- TODO: 本当は、このGADsの出力はTensorであるべき。また、TensorにIRの情報を持たせるべき
data FrontendIR where
    Add :: Tensor -> Tensor -> FrontendIR
    Sub :: Tensor -> Tensor -> FrontendIR
    Mul :: Tensor -> Tensor -> FrontendIR
    Div :: Tensor -> Tensor -> FrontendIR

opStr :: FrontendIR -> String
opStr = \case
    Add _ _ -> "+"
    Sub _ _ -> "-"
    Mul _ _ -> "*"
    Div _ _ -> "/"

gridDim :: FrontendIR -> Int
gridDim (Add (Tensor shape _) _) = shapeIdx shape 0
gridDim (Sub (Tensor shape _) _) = shapeIdx shape 0
gridDim (Mul (Tensor shape _) _) = shapeIdx shape 0
gridDim (Div (Tensor shape _) _) = shapeIdx shape 0
