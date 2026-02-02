{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE RecordWildCards       #-}

module Polyhedral.Parse where

import           ISL              (ISL, unionMap, unionMapIntersectDomain,
                                   unionSet)
import           Polyhedral.Types (Access (..), Domain (..),
                                   PolyhedralModel (..), ProgramOrder (..))

data RawPolyhedralModel = RawPolyhedralModel
    { domain          :: String
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
    dom <- Domain <$> unionSet domain
    po <- unionMap programOrder
    ra <- bound readAccess dom
    wa <- bound writeAccess dom
    rd <- Domain <$> unionSet reductionDomain
    rr <- bound reductionRead rd
    rw <- bound reductionWrite rd
    return
        PolyhedralModel
            { domain = dom
            , programOrder = ProgramOrder po
            , readAccess = ra
            , writeAccess = wa
            , reductionDomain = rd
            , reductionRead = rr
            , reductionWrite = rw
            }
