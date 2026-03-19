module Polyhedral.Internal.Set.Types (
    Set (..),
    UnionSet (..),
) where

import           Foreign.ForeignPtr      (ForeignPtr)
import           Polyhedral.Internal.FFI (IslSet, IslUnionSet)

newtype Set s = Set (ForeignPtr IslSet)

newtype UnionSet s = UnionSet (ForeignPtr IslUnionSet)
