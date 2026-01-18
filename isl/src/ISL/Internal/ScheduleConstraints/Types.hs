module ISL.Internal.ScheduleConstraints.Types (
    ScheduleConstraints (..),
) where

import           Foreign.ForeignPtr (ForeignPtr)
import           ISL.Internal.FFI   (IslScheduleConstraints)

newtype ScheduleConstraints s = ScheduleConstraints (ForeignPtr IslScheduleConstraints)
