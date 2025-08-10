{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase            #-}
{-# LANGUAGE OverloadedRecordDot   #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RecordWildCards       #-}

module FrontendIR (FrontendIR (..), elmTyToStr, EOpTy (..), Op (..), FTensor (..), codegen) where

import           Data.List (intercalate)

elmTyToStr :: EOpTy -> String
elmTyToStr Add = "+"
elmTyToStr Sub = "-"
elmTyToStr Mul = "*"
elmTyToStr Div = "/"

data EOpTy = Add | Mul | Div | Sub

data Op = ElementWise {name :: String, a :: String, b :: String, c :: String, ty :: EOpTy}

data FTensor = FTensor {name :: String, shape :: [Int]}

data FrontendIR = FrontendIR {tensors :: [FTensor], inputs :: [String], outputs :: [String], ops :: [Op]}

getInputsName :: FrontendIR -> [String]
getInputsName FrontendIR{inputs} = inputs

getOutputsName :: FrontendIR -> [String]
getOutputsName FrontendIR{outputs} = outputs

codegen :: FrontendIR -> [String]
codegen ir@FrontendIR{..} =
    let inputs = getInputsName ir
        outputs = getOutputsName ir
        a = head tensors
        aShape = a.shape
        size = head aShape
        argLists =
            (map (\name -> "float *__restrict__ " ++ name) inputs)
                ++ (map (\name -> "float *__restrict__" ++ name) outputs)
        arguments = intercalate ", " argLists
        op0 = (head ops).ty
        opStr = elmTyToStr op0
     in [ "extern \"C\" __global__ void kernel(" ++ arguments ++ ") {"
        , "  int idx =  blockIdx.x * blockDim.x + threadIdx.x;"
        , "  if (idx < " ++ show size ++ ") c[idx] = a[idx] " ++ opStr ++ " b[idx];"
        , "}"
        ]
