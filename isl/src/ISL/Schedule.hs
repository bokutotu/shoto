{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module ISL.Schedule (
    -- * Types
    Schedule (..),

    -- * Schedule Operations
    schedule,
    scheduleToString,
    scheduleDomain,
    scheduleFromDomain,
) where

import           Control.Exception      (bracket)
import           Control.Monad.IO.Class (liftIO)
import           Data.String            (IsString (..))
import           Foreign.C.String       (peekCString, withCString)
import           Foreign.ForeignPtr     (ForeignPtr, withForeignPtr)
import           Foreign.Marshal.Alloc  (free)
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
