module Polyhedral.Empty where

import           ISL              (ISL, unionMap, unionMapIsEmpty, unionSet,
                                   unionSetIsEmpty)
import           Polyhedral.Types (FromUnionMap (fromUnionMap),
                                   FromUnionSet (fromUnionSet),
                                   IntoUnionMap (intoUnionMap),
                                   IntoUnionSet (intoUnionSet))

emptyMap :: (FromUnionMap a) => ISL s (a s)
emptyMap = fromUnionMap <$> unionMap "{}"

emptySet :: (FromUnionSet a) => ISL s (a s)
emptySet = fromUnionSet <$> unionSet "{}"

isEmptyMap :: (IntoUnionMap a) => a s -> ISL s Bool
isEmptyMap m = unionMapIsEmpty $ intoUnionMap m

isEmptySet :: (IntoUnionSet a) => a s -> ISL s Bool
isEmptySet s = unionSetIsEmpty $ intoUnionSet s
