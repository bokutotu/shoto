{-# LANGUAGE ForeignFunctionInterface #-}

module ISL.FFI (
    -- * Raw Pointer Types
    IslCtx,
    IslSet,
    IslUnionSet,
    IslSchedule,
    RawCtx,
    RawSet,
    RawUnionSet,
    RawSchedule,

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
) where

import           Foreign.C.String (CString)
import           Foreign.C.Types  (CInt (..))
import           Foreign.Ptr      (FunPtr, Ptr)

-- Raw pointer types
data IslCtx

data IslSet

data IslUnionSet

data IslSchedule

type RawCtx = Ptr IslCtx

type RawSet = Ptr IslSet

type RawUnionSet = Ptr IslUnionSet

type RawSchedule = Ptr IslSchedule

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
