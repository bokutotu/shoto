{-# LANGUAGE DerivingVia                #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Polyhedral.Types where

import           ISL (ISL, Set, UnionMap, UnionSet, unionMapIsEmpty,
                      unionSetIsEmpty)

newtype Domain s = Domain (UnionSet s)
    deriving newtype (IsEmpty, IntoUnionSet, FromUnionSet)

newtype Access t s = Access (UnionMap s)
    deriving newtype (IsEmpty, IntoUnionMap, FromUnionMap)

newtype ProgramOrder s = ProgramOrder (UnionMap s)
    deriving newtype (IsEmpty, IntoUnionMap, FromUnionMap)

newtype Dependency s = Dependency (UnionMap s)
    deriving newtype (IsEmpty, IntoUnionMap, FromUnionMap)

data WriteMap

data ReadMap

data PolyhedralModel s = PolyhedralModel
    { context :: Set s
    , domain :: Domain s
    , programOrder :: ProgramOrder s
    , readAccess :: Access ReadMap s
    , writeAccess :: Access WriteMap s
    , reductionDomain :: Domain s
    , reductionRead :: Access ReadMap s
    , reductionWrite :: Access WriteMap s
    }

{- | Scheduling constraints derived from dependences.

Currently we use the same relation for both isl's validity and coincidence.
-}
data Dependencies s = Dependencies
    { legality :: Dependency s
    , proximity :: Dependency s
    }

class Empty a where
    empty :: ISL s (a s)

class IsEmpty a where
    isEmpty :: a s -> ISL s Bool

class IntoUnionMap a where
    intoUnionMap :: a s -> UnionMap s

class IntoUnionSet a where
    intoUnionSet :: a s -> UnionSet s

class FromUnionMap a where
    fromUnionMap :: UnionMap s -> a s

class FromUnionSet a where
    fromUnionSet :: UnionSet s -> a s

instance IsEmpty UnionMap where
    isEmpty = unionMapIsEmpty

instance IsEmpty UnionSet where
    isEmpty = unionSetIsEmpty

instance IntoUnionMap UnionMap where
    intoUnionMap = id

instance IntoUnionSet UnionSet where
    intoUnionSet = id

instance FromUnionMap UnionMap where
    fromUnionMap = id

instance FromUnionSet UnionSet where
    fromUnionSet = id
