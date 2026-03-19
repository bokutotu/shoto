module Polyhedral.Internal.Set (
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

import           Polyhedral.Internal.Set.Ops
import           Polyhedral.Internal.Set.Types
