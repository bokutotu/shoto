module Polyhedral.Internal.Schedule.Types (
    Schedule (..),
) where

import           Foreign.ForeignPtr      (ForeignPtr)
import           Polyhedral.Internal.FFI (IslSchedule)

newtype Schedule s = Schedule (ForeignPtr IslSchedule)
