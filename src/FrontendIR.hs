{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase            #-}
{-# LANGUAGE OverloadedStrings     #-}

module FrontendIR where

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

data ElementWise = ElementWise {a :: Tensor, b :: Tensor, ty :: ElmTy}

data ElmTy = AddOp | SubOp | MulOp | DivOp

elmTypToStr :: ElmTy -> String
elmTypToStr AddOp = "+"
elmTypToStr SubOp = "-"
elmTypToStr MulOp = "*"
elmTypToStr DivOp = "/"

data Activation = Activation {a :: Tensor, b :: Tensor, ty :: ActiTy}

data ActiTy = Relu | Softmax

data IR = ElmWise ElementWise | Acti Activation

irToOpStr :: IR -> String
irToOpStr (ElmWise ElementWise{ty}) = elmTypToStr ty

convert :: FrontendIR -> IR
convert = \case
    Add a b -> ElmWise ElementWise{a, b, ty = AddOp}
    Sub a b -> ElmWise ElementWise{a, b, ty = SubOp}
    Mul a b -> ElmWise ElementWise{a, b, ty = MulOp}
    Div a b -> ElmWise ElementWise{a, b, ty = DivOp}

getTensorSize :: IR -> Int
getTensorSize (ElmWise ElementWise{a = Tensor{shape}}) = shapeIdx shape 0

codegen :: IR -> [String]
codegen ir =
    let op = irToOpStr ir
        size = getTensorSize ir
     in [ "extern \"C\" __global__ void kernel(float *__restrict__ a, float* __restrict__ b, float* __restrict__ c) {"
        , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        , "  if (idx < " ++ show size ++ ") c[idx] = a[idx] " ++ op ++ " b[idx];"
        , "}"
        ]
