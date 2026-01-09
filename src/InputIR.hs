{-# LANGUAGE DuplicateRecordFields #-}

module InputIR where

data Func = Func
    deriving (Show, Eq, Ord)

data Domain
    deriving (Show, Eq, Ord)

data Affine
    deriving (Show, Eq, Ord)

data Compute
    deriving (Show, Eq, Ord)

data Axis
    deriving (Show, Eq, Ord)

data Access
    deriving (Show, Eq, Ord)

data Tensor
    deriving (Show, Eq, Ord)

data IndexVar
    deriving (Show, Eq, Ord)
