module ISL.Internal.Flow.Types (
    UnionAccessInfo (..),
    UnionFlow (..),
) where

import           Foreign.ForeignPtr (ForeignPtr)
import           ISL.Internal.FFI   (IslUnionAccessInfo, IslUnionFlow)

newtype UnionAccessInfo s = UnionAccessInfo (ForeignPtr IslUnionAccessInfo)

newtype UnionFlow s = UnionFlow (ForeignPtr IslUnionFlow)
