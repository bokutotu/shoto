{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module ISL.Internal.Map.Ops (
    -- * Map Operations
    imap,
    mapToString,
    mapUnion,
    mapIntersect,
    mapSubtract,
    mapCoalesce,
    mapIsEqual,
    mapDomain,
    mapRange,
    mapReverse,
    mapApplyRange,
    mapApplyDomain,

    -- * UnionMap Operations
    unionMap,
    unionMapToString,
    unionMapUnion,
    unionMapIntersect,
    unionMapIntersectDomain,
    unionMapSubtract,
    unionMapCoalesce,
    unionMapIsEqual,
    unionMapIsEmpty,
    unionMapDomain,
    unionMapRange,
    unionMapReverse,
    unionMapApplyRange,
    unionMapApplyDomain,
    unionMapLexLt,
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
import           ISL.Internal.Map.Types (Map (..), UnionMap (..))
import           ISL.Internal.Set.Types (Set (..), UnionSet (..))

-- =========================================================
-- Map Implementation
-- =========================================================

-- | String literal support: m <- "{ [i] -> [j] : ... }"
instance IsString (ISL s (Map s)) where
    fromString = imap

imap :: String -> ISL s (Map s)
imap str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_map_read ctx cstr
    manage c_map_free "isl_map_read_from_str" mk Map

mapToString :: Map s -> ISL s String
mapToString (Map fp) = do
    cstr <- liftIO $ withForeignPtr fp c_map_to_str
    if cstr == nullPtr
        then throwISL "isl_map_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

{- | Binary operation helper (Copy-on-Write)
Copy inputs before passing to keep Haskell values immutable
-}
liftOp2Map ::
    (RawMap -> RawMap -> IO RawMap) ->
    String ->
    Map s ->
    Map s ->
    ISL s (Map s)
liftOp2Map op name (Map fa) (Map fb) = do
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_map_copy pa
                cb <- c_map_copy pb
                op ca cb
    manage c_map_free name mk Map

mapUnion :: Map s -> Map s -> ISL s (Map s)
mapUnion = liftOp2Map c_map_union "isl_map_union"

mapIntersect :: Map s -> Map s -> ISL s (Map s)
mapIntersect = liftOp2Map c_map_intersect "isl_map_intersect"

mapSubtract :: Map s -> Map s -> ISL s (Map s)
mapSubtract = liftOp2Map c_map_subtract "isl_map_subtract"

mapCoalesce :: Map s -> ISL s (Map s)
mapCoalesce (Map fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_map_copy ptr
            c_map_coalesce cptr
    manage c_map_free "isl_map_coalesce" mk Map

-- | Check if two maps are equal
mapIsEqual :: Map s -> Map s -> ISL s Bool
mapIsEqual (Map fa) (Map fb) = do
    result <- liftIO $ withForeignPtr fa $ \pa ->
        withForeignPtr fb $ \pb -> c_map_is_equal pa pb
    case result of
        -1 -> throwISL "isl_map_is_equal"
        0  -> pure False
        _  -> pure True

-- | Get the domain of a map
mapDomain :: Map s -> ISL s (Set s)
mapDomain (Map fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_map_copy ptr
            c_map_domain cptr
    manage c_set_free "isl_map_domain" mk Set

-- | Get the range of a map
mapRange :: Map s -> ISL s (Set s)
mapRange (Map fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_map_copy ptr
            c_map_range cptr
    manage c_set_free "isl_map_range" mk Set

-- | Reverse a map (swap domain and range)
mapReverse :: Map s -> ISL s (Map s)
mapReverse (Map fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_map_copy ptr
            c_map_reverse cptr
    manage c_map_free "isl_map_reverse" mk Map

-- | Apply map2 to the range of map1: map1 . map2
mapApplyRange :: Map s -> Map s -> ISL s (Map s)
mapApplyRange = liftOp2Map c_map_apply_range "isl_map_apply_range"

-- | Apply map2 to the domain of map1
mapApplyDomain :: Map s -> Map s -> ISL s (Map s)
mapApplyDomain = liftOp2Map c_map_apply_domain "isl_map_apply_domain"

-- =========================================================
-- UnionMap Implementation
-- =========================================================

instance IsString (ISL s (UnionMap s)) where
    fromString = unionMap

unionMap :: String -> ISL s (UnionMap s)
unionMap str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_umap_read ctx cstr
    manage c_umap_free "isl_union_map_read_from_str" mk UnionMap

unionMapToString :: UnionMap s -> ISL s String
unionMapToString (UnionMap fp) = do
    cstr <- liftIO $ withForeignPtr fp c_umap_to_str
    if cstr == nullPtr
        then throwISL "isl_union_map_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

liftOp2UM ::
    (RawUnionMap -> RawUnionMap -> IO RawUnionMap) ->
    String ->
    UnionMap s ->
    UnionMap s ->
    ISL s (UnionMap s)
liftOp2UM op name (UnionMap fa) (UnionMap fb) = do
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_umap_copy pa
                cb <- c_umap_copy pb
                op ca cb
    manage c_umap_free name mk UnionMap

unionMapUnion :: UnionMap s -> UnionMap s -> ISL s (UnionMap s)
unionMapUnion = liftOp2UM c_umap_union "isl_union_map_union"

unionMapIntersect :: UnionMap s -> UnionMap s -> ISL s (UnionMap s)
unionMapIntersect = liftOp2UM c_umap_intersect "isl_union_map_intersect"

-- | Intersect the domain of a union map with a union set
unionMapIntersectDomain :: UnionMap s -> UnionSet s -> ISL s (UnionMap s)
unionMapIntersectDomain (UnionMap umFP) (UnionSet usFP) = do
    let mk = withForeignPtr umFP $ \umPtr ->
            withForeignPtr usFP $ \usPtr -> do
                umCopy <- c_umap_copy umPtr
                usCopy <- c_uset_copy usPtr
                c_umap_intersect_domain umCopy usCopy
    manage c_umap_free "isl_union_map_intersect_domain" mk UnionMap

unionMapSubtract :: UnionMap s -> UnionMap s -> ISL s (UnionMap s)
unionMapSubtract = liftOp2UM c_umap_subtract "isl_union_map_subtract"

unionMapCoalesce :: UnionMap s -> ISL s (UnionMap s)
unionMapCoalesce (UnionMap fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_umap_copy ptr
            c_umap_coalesce cptr
    manage c_umap_free "isl_union_map_coalesce" mk UnionMap

-- | Check if two union maps are equal
unionMapIsEqual :: UnionMap s -> UnionMap s -> ISL s Bool
unionMapIsEqual (UnionMap fa) (UnionMap fb) = do
    result <- liftIO $ withForeignPtr fa $ \pa ->
        withForeignPtr fb $ \pb -> c_umap_is_equal pa pb
    case result of
        -1 -> throwISL "isl_union_map_is_equal"
        0  -> pure False
        _  -> pure True

-- | Check if a union map is empty
unionMapIsEmpty :: UnionMap s -> ISL s Bool
unionMapIsEmpty (UnionMap fp) = do
    result <- liftIO $ withForeignPtr fp c_umap_is_empty
    case result of
        -1 -> throwISL "isl_union_map_is_empty"
        0  -> pure False
        _  -> pure True

-- | Get the domain of a union map
unionMapDomain :: UnionMap s -> ISL s (UnionSet s)
unionMapDomain (UnionMap fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_umap_copy ptr
            c_umap_domain cptr
    manage c_uset_free "isl_union_map_domain" mk UnionSet

-- | Get the range of a union map
unionMapRange :: UnionMap s -> ISL s (UnionSet s)
unionMapRange (UnionMap fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_umap_copy ptr
            c_umap_range cptr
    manage c_uset_free "isl_union_map_range" mk UnionSet

-- | Reverse a union map (swap domain and range)
unionMapReverse :: UnionMap s -> ISL s (UnionMap s)
unionMapReverse (UnionMap fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_umap_copy ptr
            c_umap_reverse cptr
    manage c_umap_free "isl_union_map_reverse" mk UnionMap

-- | Apply umap2 to the range of umap1
unionMapApplyRange :: UnionMap s -> UnionMap s -> ISL s (UnionMap s)
unionMapApplyRange = liftOp2UM c_umap_apply_range "isl_union_map_apply_range"

-- | Apply umap2 to the domain of umap1
unionMapApplyDomain :: UnionMap s -> UnionMap s -> ISL s (UnionMap s)
unionMapApplyDomain = liftOp2UM c_umap_apply_domain "isl_union_map_apply_domain"

-- | Lexicographic less-than relation between two union maps
unionMapLexLt :: UnionMap s -> UnionMap s -> ISL s (UnionMap s)
unionMapLexLt = liftOp2UM c_umap_lex_lt_union_map "isl_union_map_lex_lt_union_map"
