{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase            #-}
{-# LANGUAGE RecordWildCards       #-}

module Polyhedral.Parse where

import           Control.Monad.Except (catchError, throwError)
import           Polyhedral.Error     (ParseError (..), PolyhedralError (..))
import           Polyhedral.Internal  (ISL, set, unionMap,
                                       unionMapIntersectDomain, unionSet,
                                       unionSetIntersect)
import           Polyhedral.Types     (Access (..), Domain (..),
                                       IntoUnionSet (intoUnionSet),
                                       PolyhedralModel (..), ProgramOrder (..))

data RawPolyhedralModel = RawPolyhedralModel
    { context :: String
    , domain :: String
    , programOrder :: String
    , readAccess :: String
    , writeAccess :: String
    , reductionDomain :: String
    , reductionRead :: String
    , reductionWrite :: String
    }
    deriving (Show, Eq)

bound :: String -> Domain s -> ISL s (Access t s)
bound a (Domain dom) = Access <$> (unionMap a >>= flip unionMapIntersectDomain dom)

parsePolyhedralModel :: RawPolyhedralModel -> ISL s (PolyhedralModel s)
parsePolyhedralModel RawPolyhedralModel{..} = do
    ctx <- withParseError ParseContext (set context)
    dom <- Domain <$> withParseError ParseDomain (unionSet domain)
    po <-
        ProgramOrder
            <$> withParseError
                ParseProgramOrder
                (unionMap programOrder >>= flip unionMapIntersectDomain (intoUnionSet dom))
    ra <- withParseError ParseReadAccess (bound readAccess dom)
    wa <- withParseError ParseWriteAccess (bound writeAccess dom)
    rd <-
        Domain
            <$> withParseError
                ParseReductionDomain
                (unionSet reductionDomain >>= flip unionSetIntersect (intoUnionSet dom))
    rr <- withParseError ParseReductionRead (bound reductionRead rd)
    rw <- withParseError ParseReductionWrite (bound reductionWrite rd)
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

withParseError :: ParseError -> ISL s a -> ISL s a
withParseError parseErr action =
    catchError
        action
        ( \case
            InternalIslError islErr ->
                throwError (PolyhedralParseError parseErr (Just islErr))
            other -> throwError other
        )
