module Polyhedral.Internal.Map (
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
    unionMapIsEmpty,
    unionMapDomain,
    unionMapRange,
    unionMapReverse,
    unionMapApplyRange,
    unionMapApplyDomain,
    unionMapLexLt,
) where

import           Polyhedral.Internal.Map.Ops
import           Polyhedral.Internal.Map.Types
