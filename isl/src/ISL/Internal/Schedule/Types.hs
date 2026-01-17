module ISL.Internal.Schedule.Types (
    Schedule (..),
) where

import           Foreign.ForeignPtr (ForeignPtr)
import           ISL.Internal.FFI   (IslSchedule)

newtype Schedule s = Schedule (ForeignPtr IslSchedule)
