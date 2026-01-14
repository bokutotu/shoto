{-# OPTIONS_GHC -Wno-orphans #-}

module ISL.Ast (
    module ISL.Ast.Types,
    setExprToString,
    unionSetExprToString,
    mapExprToString,
    unionMapExprToString,
    affineExprToString,
    multiAffineExprToString,
    scheduleTreeToString,
    constraintToString,
    parseSetExpr,
    parseUnionSetExpr,
    parseMapExpr,
    parseUnionMapExpr,
    parseAffineExpr,
    parseMultiAffineExpr,
    parseScheduleTree,
    parseConstraint,
) where

import           ISL.Ast.Read  (parseAffineExpr, parseConstraint, parseMapExpr,
                                parseMultiAffineExpr, parseScheduleTree,
                                parseSetExpr, parseUnionMapExpr,
                                parseUnionSetExpr)
import           ISL.Ast.Show  (affineExprToString, constraintToString,
                                mapExprToString, multiAffineExprToString,
                                scheduleTreeToString, setExprToString,
                                unionMapExprToString, unionSetExprToString)
import           ISL.Ast.Types

instance Show SetExpr where
    show = setExprToString

instance Show UnionSetExpr where
    show = unionSetExprToString

instance Show MapExpr where
    show = mapExprToString

instance Show UnionMapExpr where
    show = unionMapExprToString

instance Show AffineExpr where
    show = affineExprToString

instance Show Constraint where
    show = constraintToString

instance Show MultiAffineExpr where
    show = multiAffineExprToString

instance Show ScheduleTree where
    show = scheduleTreeToString

instance Read SetExpr where
    readsPrec _ = readWith parseSetExpr

instance Read UnionSetExpr where
    readsPrec _ = readWith parseUnionSetExpr

instance Read MapExpr where
    readsPrec _ = readWith parseMapExpr

instance Read UnionMapExpr where
    readsPrec _ = readWith parseUnionMapExpr

instance Read AffineExpr where
    readsPrec _ = readWith parseAffineExpr

instance Read Constraint where
    readsPrec _ = readWith parseConstraint

instance Read MultiAffineExpr where
    readsPrec _ = readWith parseMultiAffineExpr

instance Read ScheduleTree where
    readsPrec _ = readWith parseScheduleTree

readWith :: (String -> Either String a) -> ReadS a
readWith parser input =
    case parser input of
        Left _       -> []
        Right result -> [(result, "")]
