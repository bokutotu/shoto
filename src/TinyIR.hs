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
    | Reduce ReduceOp [Int]

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

codegenReduce :: ReduceOp -> Shape -> [String]
codegenReduce = undefined

-- 今後はこのレイヤーではコード生成は行われないが、中間生成ぶつとしてTinyIRからコード生成を行う
-- また、TinyIRでは現在Opsは一つの場合のみを想定
-- shape validationも一切なし
codegenTiny :: TinyIR -> [String]
codegenTiny = undefined
