module ISL.Ast.Types (
    SpaceDim (..),
    spaceDim,
    DimKind (..),
    DimRef (..),
    Tuple (..),
    Space (..),
    LinearExpr (..),
    DivDef (..),
    LocalDim (..),
    LocalSpace (..),
    AffineExpr (..),
    PwAffineExpr (..),
    Relation (..),
    Constraint (..),
    BasicSet (..),
    SetExpr (..),
    UnionSetExpr (..),
    BasicMap (..),
    MapExpr (..),
    UnionMapExpr (..),
    MultiAffineExpr (..),
    MultiPwAffineExpr (..),
    MultiUnionPwAffineExpr (..),
    Band (..),
    ScheduleTree (..),
) where

import           Data.Map.Strict (Map)
import           Data.Text       (Text)

-- | Named dimension that belongs to a particular space.
newtype SpaceDim = SpaceDim {spaceDimName :: Maybe Text}
    deriving (Eq, Ord, Show)

spaceDim :: Text -> SpaceDim
spaceDim = SpaceDim . Just

data DimKind = ParamDim | InDim | OutDim | LocalDimKind
    deriving (Eq, Ord, Show)

-- | Handle that refers to a declared dimension with its original kind.
data DimRef = DimRef
    { dimKind :: DimKind
    , dimPos  :: Int
    }
    deriving (Eq, Ord, Show)

data Tuple = Tuple
    { tupleName :: Maybe Text
    , tupleDims :: [SpaceDim]
    }
    deriving (Eq, Show)

-- | ISL space, enumerating parameters, inputs, and outputs.
data Space = Space
    { spaceParams :: [SpaceDim]
    , spaceIn     :: Tuple
    , spaceOut    :: Tuple
    }
    deriving (Eq, Show)

-- | Affine form without a domain restriction (isl_aff form).
data LinearExpr = LinearExpr
    { linearConstant :: Integer
    , linearCoeffs   :: Map DimRef Integer
    }
    deriving (Eq, Show)

data DivDef = DivDef
    { divNumerator   :: LinearExpr
    , divDenominator :: Integer
    }
    deriving (Eq, Show)

data LocalDim = LocalDim
    { localDimName :: Maybe Text
    , localDimDef  :: Maybe DivDef
    }
    deriving (Eq, Show)

data LocalSpace = LocalSpace
    { localSpaceBase :: Space
    , localSpaceDims :: [LocalDim]
    }
    deriving (Eq, Show)

-- | Affine expression (isl_aff) over a local space.
data AffineExpr = AffineExpr
    { affLocalSpace :: LocalSpace
    , affForm       :: LinearExpr
    }
    deriving (Eq)

-- | Piecewise affine expression (isl_pw_aff).
data PwAffineExpr = PwAffineExpr
    { pwSpace  :: Space
    , pwPieces :: [(BasicSet, AffineExpr)]
    }
    deriving (Eq)

data Relation = RelEq | RelLe | RelGe
    deriving (Eq, Show)

data Constraint = Constraint
    { constraintRel :: Relation
    , lhsExpr       :: AffineExpr
    , rhsExpr       :: AffineExpr
    }
    deriving (Eq)

data BasicSet = BasicSet
    { basicSetSpace       :: LocalSpace
    , basicSetConstraints :: [Constraint]
    }
    deriving (Eq)

data SetExpr = SetExpr
    { setSpace :: Space
    , setParts :: [BasicSet]
    }
    deriving (Eq)

newtype UnionSetExpr = UnionSetExpr [SetExpr]
    deriving (Eq)

data BasicMap = BasicMap
    { basicMapSpace       :: LocalSpace
    , basicMapConstraints :: [Constraint]
    }
    deriving (Eq)

data MapExpr = MapExpr
    { mapSpace :: Space
    , mapParts :: [BasicMap]
    }
    deriving (Eq)

newtype UnionMapExpr = UnionMapExpr [MapExpr]
    deriving (Eq)

data MultiAffineExpr = MultiAffineExpr
    { multiAffSpace :: Space
    , multiAffExprs :: [AffineExpr]
    }
    deriving (Eq)

data MultiPwAffineExpr = MultiPwAffineExpr
    { multiPwAffSpace :: Space
    , multiPwAffExprs :: [PwAffineExpr]
    }
    deriving (Eq)

newtype MultiUnionPwAffineExpr = MultiUnionPwAffineExpr [MultiPwAffineExpr]
    deriving (Eq)

data Band = Band
    { bandSchedule        :: MultiUnionPwAffineExpr
    , bandPermutable      :: Bool
    , bandCoincident      :: [Bool]
    , bandAstBuildOptions :: Maybe UnionSetExpr
    }
    deriving (Eq)

data ScheduleTree
    = TreeBand Band [ScheduleTree]
    | TreeContext SetExpr [ScheduleTree]
    | TreeDomain UnionSetExpr [ScheduleTree]
    | TreeFilter UnionSetExpr [ScheduleTree]
    | TreeExtension UnionMapExpr [ScheduleTree]
    | TreeExpansion UnionMapExpr [ScheduleTree]
    | TreeGuard SetExpr [ScheduleTree]
    | TreeMark Text [ScheduleTree]
    | TreeSequence [ScheduleTree]
    | TreeSet [ScheduleTree]
    | TreeLeaf
    deriving (Eq)
