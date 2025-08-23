{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase            #-}

module TinyIR (
    -- Types (re-exported from TinyOps)
    Dim (..),
    Shape (..),
    Bound (..),
    SliceSpec (..),
    MovementOp (..),
    BinaryTy (..),
    UnaryTy (..),
    ElementWiseOp (..),
    ReduceOp (..),
    TinyOp (..),
    Input (..),
    TinyIR,
    -- Functions
    codegenMovement,
    codegenElementWise,
    codegenReduce,
) where

import           CoreOps               (BinaryTy (..), Bound (..), Dim (..),
                                        ElementWiseOp (..), MovementOp (..),
                                        ReduceOp (..), Shape, SliceSpec (..),
                                        TinyOp (..), UnaryTy (..))
import qualified Data.ByteString.Char8 as BS
import           IR                    (IR)

data Input = Input {shape :: Shape}

type TinyIR = IR Input TinyOp

-- 今後より低いレイヤーにlowingするので、一旦undef
codegenMovement :: MovementOp -> Shape -> BS.ByteString
codegenMovement = undefined

codegenElementWise :: ElementWiseOp -> [Shape] -> BS.ByteString
codegenElementWise op _ =
    case op of
        Binary ty ->
            BS.pack $
                unlines
                    [ "extern \"C\" __global__ void kernel(float *__restrict__ a, float *__restrict__ b, float *__restrict__ out, int n) {"
                    , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
                    , "  if (idx < n) {"
                    , "    out[idx] = a[idx] " ++ opBiStr ty ++ " b[idx];"
                    , "  }"
                    , "}"
                    ]
        Unary ty ->
            BS.pack $
                unlines
                    [ "__global__ void unary_kernel(float *__restrict__ in, float *__restrict__ out, int n) {"
                    , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
                    , "  if (idx < n) {"
                    , "    out[idx] = " ++ opUnStr ty ++ "(in[idx]);"
                    , "  }"
                    , "}"
                    ]
  where
    opBiStr = \case
        Add -> "+"
        Sub -> "-"
        Mul -> "*"
        Div -> "/"
    opUnStr = \case
        Log -> "logf"
        Root -> "sqrtf"

codegenReduce :: ReduceOp -> Maybe Int -> Shape -> BS.ByteString
codegenReduce op maybeAxis shape =
    case maybeAxis of
        Nothing -> generateFullReduce op
        Just axis -> generatePartialReduce op axis shape

generateFullReduce :: ReduceOp -> BS.ByteString
generateFullReduce op =
    BS.pack $
        unlines
            [ "extern \"C\" __global__ void kernel(float *__restrict__ in, float *__restrict__ out, int n) {"
            , "  if (blockIdx.x == 0 && threadIdx.x == 0) {"
            , "    float acc = " ++ initVal op ++ ";"
            , "    for (int i = 0; i < n; i++) {"
            , "      " ++ accumStatement op "acc" "in[i]"
            , "    }"
            , "    *out = acc;"
            , "  }"
            , "}"
            ]

generatePartialReduce :: ReduceOp -> Int -> Shape -> BS.ByteString
generatePartialReduce op axis shape =
    BS.pack $
        unlines
            [ "extern \"C\" __global__ void kernel("
            , "    float *__restrict__ in,"
            , "    float *__restrict__ out,"
            , "    int outer_stride,"
            , "    int reduce_size,"
            , "    int inner_size,"
            , "    int out_size"
            , ") {"
            , "  int tid = blockIdx.x * blockDim.x + threadIdx.x;"
            , "  "
            , "  if (tid < out_size) {"
            , "    float acc = " ++ initVal op ++ ";"
            , "    int in_base = (tid / inner_size) * outer_stride + (tid % inner_size);"
            , "    "
            , "    for (int i = 0; i < reduce_size; i++) {"
            , "      " ++ accumStatement op "acc" "in[in_base + i * inner_size]"
            , "    }"
            , "    out[tid] = acc;"
            , "  }"
            , "}"
            ]

initVal :: ReduceOp -> String
initVal Sum = "0.0f"
initVal Max = "-1e38f" -- 十分小さい値
initVal Min = "1e38f" -- 十分大きい値

accumStatement :: ReduceOp -> String -> String -> String
accumStatement Sum acc val = acc ++ " = " ++ acc ++ " + " ++ val ++ ";"
accumStatement Max acc val = acc ++ " = fmaxf(" ++ acc ++ ", " ++ val ++ ");"
accumStatement Min acc val = acc ++ " = fminf(" ++ acc ++ ", " ++ val ++ ");"
