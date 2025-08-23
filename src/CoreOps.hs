module CoreOps (
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
) where

-- Core tensor/shape and operation types shared across layers

data Dim = Dyn String | Static Int
    deriving (Eq, Show)

newtype Shape = Shape [Dim]
    deriving (Eq, Show)

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

-- 現時点で、f32のみ(TODO: bf16, tf32)
data TinyOp
    = Movement MovementOp
    | ElementWise ElementWiseOp
    | Reduce ReduceOp (Maybe Int)
