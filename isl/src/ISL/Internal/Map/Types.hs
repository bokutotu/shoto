module ISL.Internal.Map.Types (
    Map (..),
    UnionMap (..),
) where

import           Foreign.ForeignPtr (ForeignPtr)
import           ISL.Internal.FFI   (IslMap, IslUnionMap)

newtype Map s = Map (ForeignPtr IslMap)

newtype UnionMap s = UnionMap (ForeignPtr IslUnionMap)
