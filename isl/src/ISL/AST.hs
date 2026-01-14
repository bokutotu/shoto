{-# OPTIONS_GHC -Wno-orphans #-}

module ISL.Ast (
    module ISL.Ast.Types,
    setExprToString,
    unionSetExprToString,
    mapExprToString,
    unionMapExprToString,
    affineExprToString,
    pwAffineExprToString,
    multiAffineExprToString,
    multiPwAffineExprToString,
    multiUnionPwAffineExprToString,
    scheduleTreeToString,
    constraintToString,
    parseSetExpr,
    parseUnionSetExpr,
    parseMapExpr,
    parseUnionMapExpr,
    parseAffineExpr,
    parsePwAffineExpr,
    parseMultiAffineExpr,
    parseMultiPwAffineExpr,
    parseMultiUnionPwAffineExpr,
    parseScheduleTree,
    parseConstraint,
) where

import           ISL.Ast.Read  (parseAffineExpr, parseConstraint, parseMapExpr,
                                parseMultiAffineExpr, parseMultiPwAffineExpr,
                                parseMultiUnionPwAffineExpr, parsePwAffineExpr,
                                parseScheduleTree, parseSetExpr,
                                parseUnionMapExpr, parseUnionSetExpr)
import           ISL.Ast.Show  (affineExprToString, constraintToString,
                                mapExprToString, multiAffineExprToString,
                                multiPwAffineExprToString,
                                multiUnionPwAffineExprToString,
                                pwAffineExprToString, scheduleTreeToString,
                                setExprToString, unionMapExprToString,
                                unionSetExprToString)
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

instance Show PwAffineExpr where
    show = pwAffineExprToString

instance Show Constraint where
    show = constraintToString

instance Show MultiUnionPwAffineExpr where
    show = multiUnionPwAffineExprToString

instance Show MultiAffineExpr where
    show = multiAffineExprToString

instance Show MultiPwAffineExpr where
    show = multiPwAffineExprToString

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

instance Read PwAffineExpr where
    readsPrec _ = readWith parsePwAffineExpr

instance Read Constraint where
    readsPrec _ = readWith parseConstraint

instance Read MultiUnionPwAffineExpr where
    readsPrec _ = readWith parseMultiUnionPwAffineExpr

instance Read MultiAffineExpr where
    readsPrec _ = readWith parseMultiAffineExpr

instance Read MultiPwAffineExpr where
    readsPrec _ = readWith parseMultiPwAffineExpr

instance Read ScheduleTree where
    readsPrec _ = readWith parseScheduleTree

readWith :: (String -> Either String a) -> ReadS a
readWith parser input =
    case parser input of
        Left _       -> []
        Right result -> [(result, "")]
