module ISL.Internal.ScheduleNode.Types (
    ScheduleNode (..),
    ScheduleNodeType (..),
    scheduleNodeTypeFromCInt,
) where

import           Foreign.C.Types    (CInt)
import           Foreign.ForeignPtr (ForeignPtr)
import           ISL.Internal.FFI   (IslScheduleNode)

newtype ScheduleNode s = ScheduleNode (ForeignPtr IslScheduleNode)

data ScheduleNodeType
    = ScheduleNodeBand
    | ScheduleNodeContext
    | ScheduleNodeDomain
    | ScheduleNodeExpansion
    | ScheduleNodeExtension
    | ScheduleNodeFilter
    | ScheduleNodeLeaf
    | ScheduleNodeGuard
    | ScheduleNodeMark
    | ScheduleNodeSequence
    | ScheduleNodeSet
    | ScheduleNodeUnknown Int
    deriving (Show, Eq)

scheduleNodeTypeFromCInt :: CInt -> ScheduleNodeType
scheduleNodeTypeFromCInt t =
    case t of
        0  -> ScheduleNodeBand
        1  -> ScheduleNodeContext
        2  -> ScheduleNodeDomain
        3  -> ScheduleNodeExpansion
        4  -> ScheduleNodeExtension
        5  -> ScheduleNodeFilter
        6  -> ScheduleNodeLeaf
        7  -> ScheduleNodeGuard
        8  -> ScheduleNodeMark
        9  -> ScheduleNodeSequence
        10 -> ScheduleNodeSet
        _  -> ScheduleNodeUnknown (fromIntegral t)
