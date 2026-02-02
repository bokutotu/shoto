module ISL.Set (
    -- * Types
    Set (..),
    UnionSet (..),

    -- * Set Operations
    set,
    setToString,
    setUnion,
    setIntersect,
    setSubtract,
    setCoalesce,
    setIsEqual,

    -- * UnionSet Operations
    unionSet,
    unionSetToString,
    unionSetUnion,
    unionSetIntersect,
    unionSetSubtract,
    unionSetCoalesce,
    unionSetIsEqual,
    unionSetIsEmpty,
) where

import           ISL.Internal.Set.Ops
import           ISL.Internal.Set.Types
