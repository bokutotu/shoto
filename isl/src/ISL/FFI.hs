{-# LANGUAGE ForeignFunctionInterface #-}

module ISL.FFI (
    -- * Raw Pointer Types
    IslCtx,
    IslSet,
    IslUnionSet,
    IslSchedule,
    Ctx,
    Set,
    UnionSet,
    Schedule,

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

type Ctx = Ptr IslCtx

type Set = Ptr IslSet

type UnionSet = Ptr IslUnionSet

type Schedule = Ptr IslSchedule

-- Context
foreign import ccall "isl/ctx.h isl_ctx_alloc"
    c_ctx_alloc :: IO Ctx

foreign import ccall "isl/ctx.h &isl_ctx_free"
    p_ctx_free :: FunPtr (Ctx -> IO ())

foreign import ccall "isl/ctx.h isl_ctx_last_error_msg"
    c_ctx_last_error_msg :: Ctx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_file"
    c_ctx_last_error_file :: Ctx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_line"
    c_ctx_last_error_line :: Ctx -> IO CInt

-- Set
foreign import ccall "isl/set.h isl_set_read_from_str"
    c_set_read :: Ctx -> CString -> IO Set

foreign import ccall "isl/set.h isl_set_to_str"
    c_set_to_str :: Set -> IO CString

foreign import ccall "isl/set.h isl_set_free"
    c_set_free :: Set -> IO ()

foreign import ccall "isl/set.h isl_set_copy"
    c_set_copy :: Set -> IO Set

foreign import ccall "isl/set.h isl_set_union"
    c_set_union :: Set -> Set -> IO Set

foreign import ccall "isl/set.h isl_set_intersect"
    c_set_intersect :: Set -> Set -> IO Set

foreign import ccall "isl/set.h isl_set_subtract"
    c_set_subtract :: Set -> Set -> IO Set

foreign import ccall "isl/set.h isl_set_coalesce"
    c_set_coalesce :: Set -> IO Set

-- Union Set
foreign import ccall "isl/union_set.h isl_union_set_read_from_str"
    c_uset_read :: Ctx -> CString -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_to_str"
    c_uset_to_str :: UnionSet -> IO CString

foreign import ccall "isl/union_set.h isl_union_set_free"
    c_uset_free :: UnionSet -> IO ()

foreign import ccall "isl/union_set.h isl_union_set_copy"
    c_uset_copy :: UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_union"
    c_uset_union :: UnionSet -> UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_intersect"
    c_uset_intersect :: UnionSet -> UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_subtract"
    c_uset_subtract :: UnionSet -> UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_coalesce"
    c_uset_coalesce :: UnionSet -> IO UnionSet

-- Schedule
foreign import ccall "isl/schedule.h isl_schedule_read_from_str"
    c_sched_read :: Ctx -> CString -> IO Schedule

foreign import ccall "isl/schedule.h isl_schedule_to_str"
    c_sched_to_str :: Schedule -> IO CString

foreign import ccall "isl/schedule.h isl_schedule_free"
    c_sched_free :: Schedule -> IO ()

foreign import ccall "isl/schedule.h isl_schedule_from_domain"
    c_sched_from_domain :: UnionSet -> IO Schedule

foreign import ccall "isl/schedule.h isl_schedule_get_domain"
    c_sched_get_domain :: Schedule -> IO UnionSet
