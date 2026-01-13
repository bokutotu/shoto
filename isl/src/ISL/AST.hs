module ISL.Ast (
    SpaceDim (..),
    spaceDim,
    DimKind (..),
    DimRef (..),
    Space (..),
    LinearExpr (..),
    AffineExpr (..),
    Relation (..),
    Constraint (..),
    SetExpr (..),
    UnionSetExpr (..),
    MapExpr (..),
    UnionMapExpr (..),
    MultiAffineExpr (..),
    ScheduleTree (..),
) where

import           Data.Map.Strict (Map)
import           Data.Text       (Text)

-- | Named dimension that belongs to a particular space.
newtype SpaceDim = SpaceDim {spaceDimName :: Text}
    deriving (Eq, Ord, Show)

spaceDim :: Text -> SpaceDim
spaceDim = SpaceDim

data DimKind = ParamDim | InDim | OutDim
    deriving (Eq, Ord, Show)

-- | Handle that refers to a declared dimension with its original kind.
data DimRef = DimRef
    { dimKind   :: DimKind
    , dimTarget :: SpaceDim
    }
    deriving (Eq, Ord, Show)

-- | ISL space, enumerating parameters, inputs, and outputs.
data Space = Space
    { spaceName    :: Maybe Text
    , spaceParams  :: [SpaceDim]
    , spaceInputs  :: [SpaceDim]
    , spaceOutputs :: [SpaceDim]
    }
    deriving (Eq, Show)

-- | Affine expression (isl_aff) over a single space.
data LinearExpr = LinearExpr
    { linearSpace :: Space
    , constant    :: Rational
    , coeffs      :: Map DimRef Rational
    }
    deriving (Eq, Show)

-- | Piecewise affine expression (isl_pw_aff / isl_multi_pw_aff).
data AffineExpr
    = AffineLinear LinearExpr
    | AffinePiecewise Space [(SetExpr, LinearExpr)]
    deriving (Eq, Show)

data Relation = RelEq | RelLe | RelGe
    deriving (Eq, Show)

data Constraint = Constraint
    { constraintRel :: Relation
    , lhsExpr       :: AffineExpr
    , rhsExpr       :: AffineExpr
    }
    deriving (Eq, Show)

data SetExpr = SetExpr
    { setSpace       :: Space
    , setConstraints :: [Constraint]
    }
    deriving (Eq, Show)

newtype UnionSetExpr = UnionSetExpr [SetExpr]
    deriving (Eq, Show)

data MapExpr = MapExpr
    { mapDomain      :: Space
    , mapRange       :: Space
    , mapConstraints :: [Constraint]
    }
    deriving (Eq, Show)

newtype UnionMapExpr = UnionMapExpr [MapExpr]
    deriving (Eq, Show)

data MultiAffineExpr = MultiAffineExpr
    { multiSpace :: Space
    , multiExprs :: [AffineExpr]
    }
    deriving (Eq, Show)

data ScheduleTree
    = TreeBand MultiAffineExpr Bool [ScheduleTree]
    | TreeContext SetExpr [ScheduleTree]
    | TreeDomain UnionSetExpr [ScheduleTree]
    | TreeFilter UnionSetExpr [ScheduleTree]
    | TreeExtension UnionMapExpr [ScheduleTree]
    | TreeGuard SetExpr [ScheduleTree]
    | TreeSequence [ScheduleTree]
    | TreeSet [ScheduleTree]
    | TreeLeaf
    deriving (Eq, Show)
