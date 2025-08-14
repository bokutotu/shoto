{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase            #-}

module TinyIR where

import qualified Data.Set as S
import           Graph    (Graph, Node (..), filterGraph)

data Dim = Dyn String | Static Int

type Shape = [Dim]

data Bound = IntBound Int | SymBound String

data SliceSpec = SliceSpec {axis :: Int, start :: Bound, stop :: Bound, step :: Int}

data MovementOp
    = Reshape Shape
    | Permute [Int]
    | Expand Shape
    | Pad [(Int, Int)] -- (start pad, end pad)
    | Slice [SliceSpec]

data BinaryTy = Add | Sub | Mul | Div

data UnaryTy = Log | Root

data ElementWiseOp
    = Binary BinaryTy
    | Unary UnaryTy

data ReduceOp = Sum | Max | Min

-- TensorRefがない理由は、Refしたい演算の出力ノードからエッジを引くことで表現できるから(できるよね？要件等)
-- Constはより上位レイヤーから渡されるはず??あっているよね
-- 現時点で、f32のみ(TODO: bf16, tf32)
data TinyOp
    = Movement MovementOp
    | ElementWise ElementWiseOp
    | Reduce ReduceOp (Maybe Int)

data TinyNode = Input Shape | Op TinyOp

type TinyIR = Graph TinyNode

filterOps :: TinyIR -> S.Set (Node TinyNode)
filterOps = filterGraph inner
  where
    inner (Op _) = True
    inner _ = False

codegenMovement :: MovementOp -> Shape -> [String]
codegenMovement = undefined

codegenElementWise :: ElementWiseOp -> [Shape] -> [String]
codegenElementWise op _ =
    case op of
        Binary ty ->
            [ "extern \"C\" __global__ void kernel(float* a, float* b, float* out, int n) {"
            , "  int idx = blockIdx.x * blockDim.x + threadIdx.x;"
            , "  if (idx < n) {"
            , "    out[idx] = a[idx] " ++ opBiStr ty ++ " b[idx];"
            , "  }"
            , "}"
            ]
        Unary ty ->
            [ "__global__ void unary_kernel(float* in, float* out, int n) {"
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

codegenReduce :: ReduceOp -> Maybe Int -> Shape -> [String]
codegenReduce op maybeAxis shape =
    case maybeAxis of
        Nothing -> generateFullReduce op
        Just axis -> generatePartialReduce op axis shape

generateFullReduce :: ReduceOp -> [String]
generateFullReduce op =
    [ "extern \"C\" __global__ void kernel(float* in, float* out, int n) {"
    , "  if (blockIdx.x == 0 && threadIdx.x == 0) {"
    , "    float acc = " ++ initVal op ++ ";"
    , "    for (int i = 0; i < n; i++) {"
    , "      " ++ accumStatement op "acc" "in[i]"
    , "    }"
    , "    *out = acc;"
    , "  }"
    , "}"
    ]

generatePartialReduce :: ReduceOp -> Int -> Shape -> [String]
generatePartialReduce op axis shape =
    [ "extern \"C\" __global__ void kernel("
    , "    float* in,"
    , "    float* out,"
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

-- 今後はこのレイヤーではコード生成は行われないが、中間生成ぶつとしてTinyIRからコード生成を行う
-- また、TinyIRでは現在Opsは一つの場合のみを想定
-- shape validationも一切なし
codegenTiny :: TinyIR -> [String]
codegenTiny = undefined
