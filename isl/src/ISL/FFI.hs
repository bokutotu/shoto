{-# LANGUAGE ForeignFunctionInterface #-}

module ISL.FFI (
    -- * Raw Pointer Types
    IslCtx,
    IslSet,
    IslUnionSet,
    IslSchedule,
    IslScheduleNode,
    IslId,
    IslUnionMap,
    IslMultiUnionPwAff,
    RawCtx,
    RawSet,
    RawUnionSet,
    RawSchedule,
    RawScheduleNode,
    RawId,
    RawUnionMap,
    RawMultiUnionPwAff,

    -- * Context FFI
    c_ctx_alloc,
    p_ctx_free,
    c_ctx_last_error_msg,
    c_ctx_last_error_file,
    c_ctx_last_error_line,

    -- * Set FFI
    c_set_read,
    c_set_to_str,
    c_set_free,
    c_set_copy,
    c_set_union,
    c_set_intersect,
    c_set_subtract,
    c_set_coalesce,

    -- * Union Set FFI
    c_uset_read,
    c_uset_to_str,
    c_uset_free,
    c_uset_copy,
    c_uset_union,
    c_uset_intersect,
    c_uset_subtract,
    c_uset_coalesce,

    -- * Schedule FFI
    c_sched_read,
    c_sched_to_str,
    c_sched_free,
    c_sched_from_domain,
    c_sched_get_domain,

    -- * Schedule Node FFI
    c_sched_get_root,
    c_sched_node_free,
    c_sched_node_copy,
    c_sched_node_get_type,
    c_sched_node_n_children,
    c_sched_node_get_child,

    -- * Schedule Node Data Getters
    c_sched_node_band_get_partial_schedule,
    c_sched_node_band_get_permutable,
    c_sched_node_band_n_member,
    c_sched_node_context_get_context,
    c_sched_node_domain_get_domain,
    c_sched_node_filter_get_filter,
    c_sched_node_guard_get_guard,
    c_sched_node_mark_get_id,
    c_sched_node_extension_get_extension,

    -- * ID FFI
    c_id_free,
    c_id_get_name,

    -- * Union Map FFI
    c_umap_to_str,
    c_umap_free,

    -- * Multi Union Pw Aff FFI
    c_mupa_to_str,
    c_mupa_free,

    -- * Schedule Node Type Constants
    nodeTypeBand,
    nodeTypeContext,
    nodeTypeDomain,
    nodeTypeExpansion,
    nodeTypeExtension,
    nodeTypeFilter,
    nodeTypeGuard,
    nodeTypeLeaf,
    nodeTypeMark,
    nodeTypeSequence,
    nodeTypeSet,
) where

import           Foreign.C.String (CString)
import           Foreign.C.Types  (CInt (..))
import           Foreign.Ptr      (FunPtr, Ptr)

-- Raw pointer types
data IslCtx

data IslSet

data IslUnionSet

data IslSchedule

data IslScheduleNode

data IslId

data IslUnionMap

data IslMultiUnionPwAff

type RawCtx = Ptr IslCtx

type RawSet = Ptr IslSet

type RawUnionSet = Ptr IslUnionSet

type RawSchedule = Ptr IslSchedule

type RawScheduleNode = Ptr IslScheduleNode

type RawId = Ptr IslId

type RawUnionMap = Ptr IslUnionMap

type RawMultiUnionPwAff = Ptr IslMultiUnionPwAff

-- Schedule Node Type Constants (isl_schedule_node_type enum)
-- From isl/schedule_type.h:
-- isl_schedule_node_error = -1
-- isl_schedule_node_band = 0
-- isl_schedule_node_context = 1
-- isl_schedule_node_domain = 2
-- ...
nodeTypeBand, nodeTypeContext, nodeTypeDomain, nodeTypeExpansion :: CInt
nodeTypeExtension, nodeTypeFilter, nodeTypeLeaf, nodeTypeGuard :: CInt
nodeTypeMark, nodeTypeSequence, nodeTypeSet :: CInt
nodeTypeBand = 0
nodeTypeContext = 1
nodeTypeDomain = 2
nodeTypeExpansion = 3

nodeTypeExtension = 4

nodeTypeFilter = 5

nodeTypeLeaf = 6

nodeTypeGuard = 7

nodeTypeMark = 8

nodeTypeSequence = 9

nodeTypeSet = 10

-- Context
foreign import ccall "isl/ctx.h isl_ctx_alloc"
    c_ctx_alloc :: IO RawCtx

foreign import ccall "isl/ctx.h &isl_ctx_free"
    p_ctx_free :: FunPtr (RawCtx -> IO ())

foreign import ccall "isl/ctx.h isl_ctx_last_error_msg"
    c_ctx_last_error_msg :: RawCtx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_file"
    c_ctx_last_error_file :: RawCtx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_line"
    c_ctx_last_error_line :: RawCtx -> IO CInt

-- Set
foreign import ccall "isl/set.h isl_set_read_from_str"
    c_set_read :: RawCtx -> CString -> IO RawSet

foreign import ccall "isl/set.h isl_set_to_str"
    c_set_to_str :: RawSet -> IO CString

foreign import ccall "isl/set.h isl_set_free"
    c_set_free :: RawSet -> IO ()

foreign import ccall "isl/set.h isl_set_copy"
    c_set_copy :: RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_union"
    c_set_union :: RawSet -> RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_intersect"
    c_set_intersect :: RawSet -> RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_subtract"
    c_set_subtract :: RawSet -> RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_coalesce"
    c_set_coalesce :: RawSet -> IO RawSet

-- Union Set
foreign import ccall "isl/union_set.h isl_union_set_read_from_str"
    c_uset_read :: RawCtx -> CString -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_to_str"
    c_uset_to_str :: RawUnionSet -> IO CString

foreign import ccall "isl/union_set.h isl_union_set_free"
    c_uset_free :: RawUnionSet -> IO ()

foreign import ccall "isl/union_set.h isl_union_set_copy"
    c_uset_copy :: RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_union"
    c_uset_union :: RawUnionSet -> RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_intersect"
    c_uset_intersect :: RawUnionSet -> RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_subtract"
    c_uset_subtract :: RawUnionSet -> RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_coalesce"
    c_uset_coalesce :: RawUnionSet -> IO RawUnionSet

-- Schedule
foreign import ccall "isl/schedule.h isl_schedule_read_from_str"
    c_sched_read :: RawCtx -> CString -> IO RawSchedule

foreign import ccall "isl/schedule.h isl_schedule_to_str"
    c_sched_to_str :: RawSchedule -> IO CString

foreign import ccall "isl/schedule.h isl_schedule_free"
    c_sched_free :: RawSchedule -> IO ()

foreign import ccall "isl/schedule.h isl_schedule_from_domain"
    c_sched_from_domain :: RawUnionSet -> IO RawSchedule

foreign import ccall "isl/schedule.h isl_schedule_get_domain"
    c_sched_get_domain :: RawSchedule -> IO RawUnionSet

-- Schedule Node
foreign import ccall "isl/schedule.h isl_schedule_get_root"
    c_sched_get_root :: RawSchedule -> IO RawScheduleNode

foreign import ccall "isl/schedule_node.h isl_schedule_node_free"
    c_sched_node_free :: RawScheduleNode -> IO ()

foreign import ccall "isl/schedule_node.h isl_schedule_node_copy"
    c_sched_node_copy :: RawScheduleNode -> IO RawScheduleNode

foreign import ccall "isl/schedule_node.h isl_schedule_node_get_type"
    c_sched_node_get_type :: RawScheduleNode -> IO CInt

foreign import ccall "isl/schedule_node.h isl_schedule_node_n_children"
    c_sched_node_n_children :: RawScheduleNode -> IO CInt

foreign import ccall "isl/schedule_node.h isl_schedule_node_get_child"
    c_sched_node_get_child :: RawScheduleNode -> CInt -> IO RawScheduleNode

-- Schedule Node Band Operations
foreign import ccall "isl/schedule_node.h isl_schedule_node_band_get_partial_schedule"
    c_sched_node_band_get_partial_schedule :: RawScheduleNode -> IO RawMultiUnionPwAff

foreign import ccall "isl/schedule_node.h isl_schedule_node_band_get_permutable"
    c_sched_node_band_get_permutable :: RawScheduleNode -> IO CInt

foreign import ccall "isl/schedule_node.h isl_schedule_node_band_n_member"
    c_sched_node_band_n_member :: RawScheduleNode -> IO CInt

-- Schedule Node Data Getters
foreign import ccall "isl/schedule_node.h isl_schedule_node_context_get_context"
    c_sched_node_context_get_context :: RawScheduleNode -> IO RawSet

foreign import ccall "isl/schedule_node.h isl_schedule_node_domain_get_domain"
    c_sched_node_domain_get_domain :: RawScheduleNode -> IO RawUnionSet

foreign import ccall "isl/schedule_node.h isl_schedule_node_filter_get_filter"
    c_sched_node_filter_get_filter :: RawScheduleNode -> IO RawUnionSet

foreign import ccall "isl/schedule_node.h isl_schedule_node_guard_get_guard"
    c_sched_node_guard_get_guard :: RawScheduleNode -> IO RawSet

foreign import ccall "isl/schedule_node.h isl_schedule_node_mark_get_id"
    c_sched_node_mark_get_id :: RawScheduleNode -> IO RawId

foreign import ccall "isl/schedule_node.h isl_schedule_node_extension_get_extension"
    c_sched_node_extension_get_extension :: RawScheduleNode -> IO RawUnionMap

-- ID Operations
foreign import ccall "isl/id.h isl_id_free"
    c_id_free :: RawId -> IO ()

foreign import ccall "isl/id.h isl_id_get_name"
    c_id_get_name :: RawId -> IO CString

-- Union Map Operations
foreign import ccall "isl/union_map.h isl_union_map_to_str"
    c_umap_to_str :: RawUnionMap -> IO CString

foreign import ccall "isl/union_map.h isl_union_map_free"
    c_umap_free :: RawUnionMap -> IO ()

-- Multi Union Pw Aff Operations
foreign import ccall "isl/aff.h isl_multi_union_pw_aff_to_str"
    c_mupa_to_str :: RawMultiUnionPwAff -> IO CString

foreign import ccall "isl/aff.h isl_multi_union_pw_aff_free"
    c_mupa_free :: RawMultiUnionPwAff -> IO ()
