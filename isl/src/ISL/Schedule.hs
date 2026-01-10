{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiWayIf        #-}
{-# LANGUAGE OverloadedStrings #-}

module ISL.Schedule (
    -- * Types
    Schedule (..),
    ScheduleNode (..),
    ScheduleTree (..),
    BandInfo (..),

    -- * Schedule Operations
    schedule,
    scheduleToString,
    scheduleDomain,
    scheduleFromDomain,

    -- * Schedule Tree Traversal
    scheduleRoot,
    scheduleTree,
) where

import           Control.Exception      (bracket)
import           Control.Monad.IO.Class (liftIO)
import           Data.String            (IsString (..))
import           Foreign.C.String       (peekCString, withCString)
import           Foreign.ForeignPtr     (ForeignPtr, withForeignPtr)
import           Foreign.Marshal.Alloc  (free)
import           Foreign.Marshal.Utils  (toBool)
import           Foreign.Ptr            (nullPtr)
import           ISL.Core
import           ISL.Set                (UnionSet (..))

newtype Schedule s = Schedule (ForeignPtr IslSchedule)

instance IsString (ISL s (Schedule s)) where
    fromString = schedule

schedule :: String -> ISL s (Schedule s)
schedule str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_sched_read ctx cstr
    manage c_sched_free "isl_schedule_read_from_str" mk Schedule

scheduleToString :: Schedule s -> ISL s String
scheduleToString (Schedule fp) = do
    cstr <- liftIO $ withForeignPtr fp c_sched_to_str
    if cstr == nullPtr
        then throwISL "isl_schedule_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

scheduleDomain :: Schedule s -> ISL s (UnionSet s)
scheduleDomain (Schedule fp) = do
    let mk = withForeignPtr fp $ \ptr -> c_sched_get_domain ptr
    manage c_uset_free "isl_schedule_get_domain" mk UnionSet

scheduleFromDomain :: UnionSet s -> ISL s (Schedule s)
scheduleFromDomain (UnionSet fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_uset_copy ptr
            c_sched_from_domain cptr
    manage c_sched_free "isl_schedule_from_domain" mk Schedule

newtype ScheduleNode s = ScheduleNode (ForeignPtr IslScheduleNode)

data BandInfo = BandInfo
    { bandSchedule   :: String
    , bandPermutable :: Bool
    , bandMembers    :: Int
    }
    deriving (Show, Eq)

data ScheduleTree
    = TreeBand BandInfo [ScheduleTree]
    | TreeContext String [ScheduleTree]
    | TreeDomain String [ScheduleTree]
    | TreeFilter String [ScheduleTree]
    | TreeGuard String [ScheduleTree]
    | TreeMark String [ScheduleTree]
    | TreeExtension String [ScheduleTree]
    | TreeSequence [ScheduleTree]
    | TreeSet [ScheduleTree]
    | TreeExpansion [ScheduleTree]
    | TreeLeaf
    | TreeUnknown Int [ScheduleTree]
    deriving (Show, Eq)

scheduleRoot :: Schedule s -> ISL s (ScheduleNode s)
scheduleRoot (Schedule schedFP) = do
    let mk = withForeignPtr schedFP c_sched_get_root
    manage c_sched_node_free "isl_schedule_get_root" mk ScheduleNode

scheduleTree :: Schedule s -> ISL s ScheduleTree
scheduleTree sched = do
    root <- scheduleRoot sched
    walkNode root

walkNode :: ScheduleNode s -> ISL s ScheduleTree
walkNode (ScheduleNode nodeFP) = do
    t <- liftIO $ withForeignPtr nodeFP c_sched_node_get_type
    n <- liftIO $ withForeignPtr nodeFP c_sched_node_n_children

    children <- mapM (\i -> getChild nodeFP (fromIntegral i) >>= walkNode) [0 .. n - 1]

    if
        | t == nodeTypeBand -> do
            info <- getBandInfo nodeFP
            pure $ TreeBand info children
        | t == nodeTypeContext -> do
            ctxStr <- getContextInfo nodeFP
            pure $ TreeContext ctxStr children
        | t == nodeTypeDomain -> do
            domStr <- getDomainInfo nodeFP
            pure $ TreeDomain domStr children
        | t == nodeTypeFilter -> do
            filtStr <- getFilterInfo nodeFP
            pure $ TreeFilter filtStr children
        | t == nodeTypeGuard -> do
            guardStr <- getGuardInfo nodeFP
            pure $ TreeGuard guardStr children
        | t == nodeTypeMark -> do
            markStr <- getMarkInfo nodeFP
            pure $ TreeMark markStr children
        | t == nodeTypeExtension -> do
            extStr <- getExtensionInfo nodeFP
            pure $ TreeExtension extStr children
        | t == nodeTypeSequence -> pure $ TreeSequence children
        | t == nodeTypeSet -> pure $ TreeSet children
        | t == nodeTypeExpansion -> pure $ TreeExpansion children
        | t == nodeTypeLeaf -> pure TreeLeaf
        | otherwise -> pure $ TreeUnknown (fromIntegral t) children
  where
    getChild :: ForeignPtr IslScheduleNode -> Int -> ISL s (ScheduleNode s)
    getChild parentFP idx = do
        let mk = withForeignPtr parentFP $ \parentPtr -> do
                parentCopy <- c_sched_node_copy parentPtr
                c_sched_node_get_child parentCopy (fromIntegral idx)
        manage c_sched_node_free "isl_schedule_node_get_child" mk ScheduleNode

    getBandInfo :: ForeignPtr IslScheduleNode -> ISL s BandInfo
    getBandInfo pNode = liftIO $ withForeignPtr pNode $ \p -> do
        mupa <- c_sched_node_band_get_partial_schedule p
        schedStr <-
            if mupa == nullPtr
                then pure ""
                else do
                    cstr <- c_mupa_to_str mupa
                    s <-
                        if cstr == nullPtr
                            then pure ""
                            else bracket (pure cstr) free peekCString
                    c_mupa_free mupa
                    pure s

        perm <- c_sched_node_band_get_permutable p
        mems <- c_sched_node_band_n_member p

        pure $
            BandInfo
                { bandSchedule = schedStr
                , bandPermutable = toBool perm
                , bandMembers = fromIntegral mems
                }

    getContextInfo :: ForeignPtr IslScheduleNode -> ISL s String
    getContextInfo pNode = liftIO $ withForeignPtr pNode $ \p -> do
        setPtr <- c_sched_node_context_get_context p
        if setPtr == nullPtr
            then pure ""
            else do
                cstr <- c_set_to_str setPtr
                str <-
                    if cstr == nullPtr
                        then pure ""
                        else bracket (pure cstr) free peekCString
                c_set_free setPtr
                pure str

    getDomainInfo :: ForeignPtr IslScheduleNode -> ISL s String
    getDomainInfo pNode = liftIO $ withForeignPtr pNode $ \p -> do
        usetPtr <- c_sched_node_domain_get_domain p
        if usetPtr == nullPtr
            then pure ""
            else do
                cstr <- c_uset_to_str usetPtr
                str <-
                    if cstr == nullPtr
                        then pure ""
                        else bracket (pure cstr) free peekCString
                c_uset_free usetPtr
                pure str

    getFilterInfo :: ForeignPtr IslScheduleNode -> ISL s String
    getFilterInfo pNode = liftIO $ withForeignPtr pNode $ \p -> do
        usetPtr <- c_sched_node_filter_get_filter p
        if usetPtr == nullPtr
            then pure ""
            else do
                cstr <- c_uset_to_str usetPtr
                str <-
                    if cstr == nullPtr
                        then pure ""
                        else bracket (pure cstr) free peekCString
                c_uset_free usetPtr
                pure str

    getGuardInfo :: ForeignPtr IslScheduleNode -> ISL s String
    getGuardInfo pNode = liftIO $ withForeignPtr pNode $ \p -> do
        setPtr <- c_sched_node_guard_get_guard p
        if setPtr == nullPtr
            then pure ""
            else do
                cstr <- c_set_to_str setPtr
                str <-
                    if cstr == nullPtr
                        then pure ""
                        else bracket (pure cstr) free peekCString
                c_set_free setPtr
                pure str

    getMarkInfo :: ForeignPtr IslScheduleNode -> ISL s String
    getMarkInfo pNode = liftIO $ withForeignPtr pNode $ \p -> do
        idPtr <- c_sched_node_mark_get_id p
        if idPtr == nullPtr
            then pure ""
            else do
                cstr <- c_id_get_name idPtr
                str <-
                    if cstr == nullPtr
                        then pure ""
                        else peekCString cstr
                c_id_free idPtr
                pure str

    getExtensionInfo :: ForeignPtr IslScheduleNode -> ISL s String
    getExtensionInfo pNode = liftIO $ withForeignPtr pNode $ \p -> do
        umapPtr <- c_sched_node_extension_get_extension p
        if umapPtr == nullPtr
            then pure ""
            else do
                cstr <- c_umap_to_str umapPtr
                str <-
                    if cstr == nullPtr
                        then pure ""
                        else bracket (pure cstr) free peekCString
                c_umap_free umapPtr
                pure str
