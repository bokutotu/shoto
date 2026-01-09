{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module ISL.Set (
    -- * Types
    SSet (..),
    SUnionSet (..),

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

    -- * UnionSet Operations
    unionSet,
    unionSetToString,
    unionSetUnion,
    unionSetIntersect,
    unionSetSubtract,
    unionSetCoalesce,
) where

import           Control.Exception      (bracket)
import           Control.Monad.IO.Class (liftIO)
import           Data.String            (IsString (..))
import           Foreign.C.String       (peekCString, withCString)
import           Foreign.ForeignPtr     (ForeignPtr, withForeignPtr)
import           Foreign.Marshal.Alloc  (free)
import           Foreign.Ptr            (nullPtr)
import           ISL.Core

-- | Type definitions (moved from Internal)
newtype SSet s = SSet (ForeignPtr IslSet)

newtype SUnionSet s = SUnionSet (ForeignPtr IslUnionSet)

-- =========================================================
-- SSet Implementation
-- =========================================================

-- | String literal support: s <- "{ ... }"
instance IsString (ISL s (SSet s)) where
    fromString = set

set :: String -> ISL s (SSet s)
set str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_set_read ctx cstr
    manage c_set_free "isl_set_read_from_str" mk SSet

setToString :: SSet s -> ISL s String
setToString (SSet fp) = do
    cstr <- liftIO $ withForeignPtr fp c_set_to_str
    if cstr == nullPtr
        then throwISL "isl_set_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

{- | Binary operation helper (Copy-on-Write)
Copy inputs before passing to keep Haskell values immutable
-}
liftOp2Set ::
    (Set -> Set -> IO Set) ->
    String ->
    SSet s ->
    SSet s ->
    ISL s (SSet s)
liftOp2Set op name (SSet fa) (SSet fb) = do
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_set_copy pa
                cb <- c_set_copy pb
                op ca cb
    manage c_set_free name mk SSet

setUnion :: SSet s -> SSet s -> ISL s (SSet s)
setUnion = liftOp2Set c_set_union "isl_set_union"

setIntersect :: SSet s -> SSet s -> ISL s (SSet s)
setIntersect = liftOp2Set c_set_intersect "isl_set_intersect"

setSubtract :: SSet s -> SSet s -> ISL s (SSet s)
setSubtract = liftOp2Set c_set_subtract "isl_set_subtract"

setCoalesce :: SSet s -> ISL s (SSet s)
setCoalesce (SSet fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_set_copy ptr
            c_set_coalesce cptr
    manage c_set_free "isl_set_coalesce" mk SSet

-- Operators
infixl 6 \/

infixl 7 /\

infixl 6 \\

(\/) :: SSet s -> SSet s -> ISL s (SSet s)
(\/) = setUnion

(/\) :: SSet s -> SSet s -> ISL s (SSet s)
(/\) = setIntersect

(\\) :: SSet s -> SSet s -> ISL s (SSet s)
(\\) = setSubtract

-- =========================================================
-- SUnionSet Implementation
-- =========================================================

instance IsString (ISL s (SUnionSet s)) where
    fromString = unionSet

unionSet :: String -> ISL s (SUnionSet s)
unionSet str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_uset_read ctx cstr
    manage c_uset_free "isl_union_set_read_from_str" mk SUnionSet

unionSetToString :: SUnionSet s -> ISL s String
unionSetToString (SUnionSet fp) = do
    cstr <- liftIO $ withForeignPtr fp c_uset_to_str
    if cstr == nullPtr
        then throwISL "isl_union_set_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

liftOp2US ::
    (UnionSet -> UnionSet -> IO UnionSet) ->
    String ->
    SUnionSet s ->
    SUnionSet s ->
    ISL s (SUnionSet s)
liftOp2US op name (SUnionSet fa) (SUnionSet fb) = do
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_uset_copy pa
                cb <- c_uset_copy pb
                op ca cb
    manage c_uset_free name mk SUnionSet

unionSetUnion :: SUnionSet s -> SUnionSet s -> ISL s (SUnionSet s)
unionSetUnion = liftOp2US c_uset_union "isl_union_set_union"

unionSetIntersect :: SUnionSet s -> SUnionSet s -> ISL s (SUnionSet s)
unionSetIntersect = liftOp2US c_uset_intersect "isl_union_set_intersect"

unionSetSubtract :: SUnionSet s -> SUnionSet s -> ISL s (SUnionSet s)
unionSetSubtract = liftOp2US c_uset_subtract "isl_union_set_subtract"

unionSetCoalesce :: SUnionSet s -> ISL s (SUnionSet s)
unionSetCoalesce (SUnionSet fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_uset_copy ptr
            c_uset_coalesce cptr
    manage c_uset_free "isl_union_set_coalesce" mk SUnionSet
