module Polyhedral.Internal.ScheduleConstraints.Types (
    ScheduleConstraints (..),
) where

import           Foreign.ForeignPtr      (ForeignPtr)
import           Polyhedral.Internal.FFI (IslScheduleConstraints)

newtype ScheduleConstraints s = ScheduleConstraints (ForeignPtr IslScheduleConstraints)
