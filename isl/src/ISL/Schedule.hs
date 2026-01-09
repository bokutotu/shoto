{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module ISL.Schedule (
    -- * Types
    SSchedule (..),

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
import           ISL.Set                (SUnionSet (..))

-- | Type definition (moved from Internal)
newtype SSchedule s = SSchedule (ForeignPtr IslSchedule)

instance IsString (ISL s (SSchedule s)) where
    fromString = schedule

schedule :: String -> ISL s (SSchedule s)
schedule str = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_sched_read ctx cstr
    manage c_sched_free "isl_schedule_read_from_str" mk SSchedule

scheduleToString :: SSchedule s -> ISL s String
scheduleToString (SSchedule fp) = do
    cstr <- liftIO $ withForeignPtr fp c_sched_to_str
    if cstr == nullPtr
        then throwISL "isl_schedule_to_str"
        else liftIO $ bracket (pure cstr) free peekCString

scheduleDomain :: SSchedule s -> ISL s (SUnionSet s)
scheduleDomain (SSchedule fp) = do
    -- get_domain returns a NEW object (+1 ref), input is kept (safe)
    let mk = withForeignPtr fp $ \ptr -> c_sched_get_domain ptr
    manage c_uset_free "isl_schedule_get_domain" mk SUnionSet

scheduleFromDomain :: SUnionSet s -> ISL s (SSchedule s)
scheduleFromDomain (SUnionSet fp) = do
    -- from_domain consumes input, so we MUST copy first
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_uset_copy ptr
            c_sched_from_domain cptr
    manage c_sched_free "isl_schedule_from_domain" mk SSchedule
