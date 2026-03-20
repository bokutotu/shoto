module Polyhedral.Error (
    IslError (..),
    ParseError (..),
    OptimizeError (..),
    PolyhedralError (..),
) where

data IslError = IslError
    { islFunction :: String
    , islMessage :: Maybe String
    , islFile :: Maybe String
    , islLine :: Maybe Int
    }
    deriving (Eq, Show)

data ParseError
    = ParseContext
    | ParseDomain
    | ParseProgramOrder
    | ParseReadAccess
    | ParseWriteAccess
    | ParseReductionDomain
    | ParseReductionRead
    | ParseReductionWrite
    deriving (Eq, Show)

data OptimizeError
    = OptimizeInternalFailure
    | OptimizeTileNoAxis
    | OptimizeTileEmptyLevel
    | OptimizeTileLevelCountMismatch
    | OptimizeTileNonPositiveSize
    | OptimizeBandRankMismatch
        { expectedRank :: Int
        , actualRank :: Int
        }
    | OptimizeTiledBandExpectedOneChild
        { actualChildren :: Int
        }
    | OptimizeTiledBandChildNotBand
    deriving (Eq, Show)

data PolyhedralError
    = InternalIslError IslError
    | PolyhedralParseError ParseError (Maybe IslError)
    | PolyhedralDependenceError (Maybe IslError)
    | PolyhedralScheduleError (Maybe IslError)
    | PolyhedralOptimizeError OptimizeError (Maybe IslError)
    | PolyhedralAstError (Maybe IslError)
    deriving (Eq, Show)
