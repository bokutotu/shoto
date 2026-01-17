module ISL.Internal.Set.Types (
    Set (..),
    UnionSet (..),
) where

import           Foreign.ForeignPtr (ForeignPtr)
import           ISL.Internal.FFI   (IslSet, IslUnionSet)

newtype Set s = Set (ForeignPtr IslSet)

newtype UnionSet s = UnionSet (ForeignPtr IslUnionSet)
