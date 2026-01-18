module ISL.Map (
    -- * Types
    Map (..),
    UnionMap (..),

    -- * Map Operations
    imap,
    mapToString,
    mapUnion,
    mapIntersect,
    mapSubtract,
    mapCoalesce,
    mapIsEqual,
    mapDomain,
    mapRange,
    mapReverse,
    mapApplyRange,
    mapApplyDomain,

    -- * UnionMap Operations
    unionMap,
    unionMapToString,
    unionMapUnion,
    unionMapIntersect,
    unionMapIntersectDomain,
    unionMapSubtract,
    unionMapCoalesce,
    unionMapIsEqual,
    unionMapDomain,
    unionMapRange,
    unionMapReverse,
    unionMapApplyRange,
    unionMapApplyDomain,
) where

import           ISL.Internal.Map.Ops
import           ISL.Internal.Map.Types
