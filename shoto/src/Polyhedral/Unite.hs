module Polyhedral.Unite where

import           ISL              (ISL, unionMapUnion, unionSetUnion)
import           Polyhedral.Types (FromUnionMap (fromUnionMap),
                                   FromUnionSet (fromUnionSet),
                                   IntoUnionMap (intoUnionMap),
                                   IntoUnionSet (intoUnionSet))

uniteMap :: (IntoUnionMap a, IntoUnionMap b, FromUnionMap c) => a s -> b s -> ISL s (c s)
uniteMap a b = do
    let ua = intoUnionMap a
        ub = intoUnionMap b
    fromUnionMap <$> (ua `unionMapUnion` ub)

uniteSet :: (IntoUnionSet a, IntoUnionSet b, FromUnionSet c) => a s -> b s -> ISL s (c s)
uniteSet a b = do
    let ua = intoUnionSet a
        ub = intoUnionSet b
    fromUnionSet <$> (ua `unionSetUnion` ub)
