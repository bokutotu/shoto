{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module ISL.Internal.Set.Ops (
    -- * Set Operations
    set,
    setToString,
    setUnion,
    (\/),
    setIntersect,
    (/\),
    setSubtract,
    (\\),
    setCoalesce,
    setIsEqual,

    -- * UnionSet Operations
    unionSet,
    unionSetToString,
    unionSetUnion,
    unionSetIntersect,
    unionSetSubtract,
    unionSetCoalesce,
    unionSetIsEqual,
) where

import           Control.Exception      (bracket)
import           Control.Monad.IO.Class (liftIO)
import           Data.String            (IsString (..))
import           Foreign.C.String       (peekCString, withCString)
import           Foreign.ForeignPtr     (withForeignPtr)
import           Foreign.Marshal.Alloc  (free)
import           Foreign.Ptr            (nullPtr)
import           ISL.Core               (Env (..), ISL, askEnv, manage,
                                         throwISL)
import           ISL.Internal.FFI
import           ISL.Internal.Set.Types (Set (..), UnionSet (..))

-- =========================================================
-- Set Implementation
-- =========================================================

-- | String literal support: s <- "{ ... }"
instance IsString (ISL s (Set s)) where
    fromString = set

set :: String -> ISL s (Set s)
set str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_set_read ctx cstr
    manage c_set_free "isl_set_read_from_str" mk Set

setToString :: Set s -> ISL s String
setToString (Set fp) = do
    cstr <- liftIO $ withForeignPtr fp c_set_to_str
    if cstr == nullPtr
        then throwISL "isl_set_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

{- | Binary operation helper (Copy-on-Write)
Copy inputs before passing to keep Haskell values immutable
-}
liftOp2Set ::
    (RawSet -> RawSet -> IO RawSet) ->
    String ->
    Set s ->
    Set s ->
    ISL s (Set s)
liftOp2Set op name (Set fa) (Set fb) = do
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_set_copy pa
                cb <- c_set_copy pb
                op ca cb
    manage c_set_free name mk Set

setUnion :: Set s -> Set s -> ISL s (Set s)
setUnion = liftOp2Set c_set_union "isl_set_union"

setIntersect :: Set s -> Set s -> ISL s (Set s)
setIntersect = liftOp2Set c_set_intersect "isl_set_intersect"

setSubtract :: Set s -> Set s -> ISL s (Set s)
setSubtract = liftOp2Set c_set_subtract "isl_set_subtract"

setCoalesce :: Set s -> ISL s (Set s)
setCoalesce (Set fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_set_copy ptr
            c_set_coalesce cptr
    manage c_set_free "isl_set_coalesce" mk Set

-- | Check if two sets are equal
setIsEqual :: Set s -> Set s -> ISL s Bool
setIsEqual (Set fa) (Set fb) = do
    result <- liftIO $ withForeignPtr fa $ \pa ->
        withForeignPtr fb $ \pb -> c_set_is_equal pa pb
    case result of
        -1 -> throwISL "isl_set_is_equal"
        0 -> pure False
        _ -> pure True

-- Operators
infixl 6 \/

infixl 7 /\

infixl 6 \\

(\/) :: Set s -> Set s -> ISL s (Set s)
(\/) = setUnion

(/\) :: Set s -> Set s -> ISL s (Set s)
(/\) = setIntersect

(\\) :: Set s -> Set s -> ISL s (Set s)
(\\) = setSubtract

-- =========================================================
-- UnionSet Implementation
-- =========================================================

instance IsString (ISL s (UnionSet s)) where
    fromString = unionSet

unionSet :: String -> ISL s (UnionSet s)
unionSet str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_uset_read ctx cstr
    manage c_uset_free "isl_union_set_read_from_str" mk UnionSet

unionSetToString :: UnionSet s -> ISL s String
unionSetToString (UnionSet fp) = do
    cstr <- liftIO $ withForeignPtr fp c_uset_to_str
    if cstr == nullPtr
        then throwISL "isl_union_set_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

liftOp2US ::
    (RawUnionSet -> RawUnionSet -> IO RawUnionSet) ->
    String ->
    UnionSet s ->
    UnionSet s ->
    ISL s (UnionSet s)
liftOp2US op name (UnionSet fa) (UnionSet fb) = do
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_uset_copy pa
                cb <- c_uset_copy pb
                op ca cb
    manage c_uset_free name mk UnionSet

unionSetUnion :: UnionSet s -> UnionSet s -> ISL s (UnionSet s)
unionSetUnion = liftOp2US c_uset_union "isl_union_set_union"

unionSetIntersect :: UnionSet s -> UnionSet s -> ISL s (UnionSet s)
unionSetIntersect = liftOp2US c_uset_intersect "isl_union_set_intersect"

unionSetSubtract :: UnionSet s -> UnionSet s -> ISL s (UnionSet s)
unionSetSubtract = liftOp2US c_uset_subtract "isl_union_set_subtract"

unionSetCoalesce :: UnionSet s -> ISL s (UnionSet s)
unionSetCoalesce (UnionSet fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_uset_copy ptr
            c_uset_coalesce cptr
    manage c_uset_free "isl_union_set_coalesce" mk UnionSet

-- | Check if two union sets are equal
unionSetIsEqual :: UnionSet s -> UnionSet s -> ISL s Bool
unionSetIsEqual (UnionSet fa) (UnionSet fb) = do
    result <- liftIO $ withForeignPtr fa $ \pa ->
        withForeignPtr fb $ \pb -> c_uset_is_equal pa pb
    case result of
        -1 -> throwISL "isl_union_set_is_equal"
        0 -> pure False
        _ -> pure True
