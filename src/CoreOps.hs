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

data Dim = Dyn String | Static Int
    deriving (Eq, Show)

newtype Shape = Shape [Dim]
    deriving (Eq, Show)

data Bound = IntBound Int | SymBound String
    deriving (Eq, Show)

data SliceSpec = SliceSpec {axis :: Int, start :: Bound, stop :: Bound, step :: Int}
    deriving (Eq, Show)

data MovementOp
    = Reshape Shape
    | Permute [Int]
    | Expand Shape
    | Pad [(Int, Int)] -- (start pad, end pad)
    | Slice [SliceSpec]
    deriving (Eq, Show)

data BinaryTy = Add | Sub | Mul | Div
    deriving (Eq, Show)

data UnaryTy = Log | Root
    deriving (Eq, Show)

data ElementWiseOp
    = Binary BinaryTy
    | Unary UnaryTy
    deriving (Eq, Show)

data ReduceOp = Sum | Max | Min
    deriving (Eq, Show)

-- 現時点で、f32のみ(TODO: bf16, tf32)
data TinyOp
    = Movement MovementOp
    | ElementWise ElementWiseOp
    | Reduce ReduceOp (Maybe Int)
