{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE RecordWildCards       #-}

module Polyhedral.Parse where

import           ISL              (ISL, set, unionMap, unionMapIntersectDomain,
                                   unionSet, unionSetIntersect)
import           Polyhedral.Types (Access (..), Domain (..),
                                   IntoUnionSet (intoUnionSet),
                                   PolyhedralModel (..), ProgramOrder (..))

data RawPolyhedralModel = RawPolyhedralModel
    { context         :: String
    , domain          :: String
    , programOrder    :: String
    , readAccess      :: String
    , writeAccess     :: String
    , reductionDomain :: String
    , reductionRead   :: String
    , reductionWrite  :: String
    }
    deriving (Show, Eq)

bound :: String -> Domain s -> ISL s (Access t s)
bound a (Domain dom) = Access <$> (unionMap a >>= flip unionMapIntersectDomain dom)

parsePolyhedralModel :: RawPolyhedralModel -> ISL s (PolyhedralModel s)
parsePolyhedralModel RawPolyhedralModel{..} = do
    ctx <- set context
    dom <- Domain <$> unionSet domain
    po <- ProgramOrder <$> (unionMap programOrder >>= flip unionMapIntersectDomain (intoUnionSet dom))
    ra <- bound readAccess dom
    wa <- bound writeAccess dom
    rd <- Domain <$> (unionSet reductionDomain >>= flip unionSetIntersect (intoUnionSet dom))
    rr <- bound reductionRead rd
    rw <- bound reductionWrite rd
    return
        PolyhedralModel
            { context = ctx
            , domain = dom
            , programOrder = po
            , readAccess = ra
            , writeAccess = wa
            , reductionDomain = rd
            , reductionRead = rr
            , reductionWrite = rw
            }
