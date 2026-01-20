{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module ISL.Internal.Schedule.Ops (
    -- * Schedule Operations
    schedule,
    scheduleToString,
    scheduleDomain,
    scheduleFromDomain,
    scheduleIsEqual,
) where

import           Control.Exception           (bracket)
import           Control.Monad.IO.Class      (liftIO)
import           Data.String                 (IsString (..))
import           Foreign.C.String            (peekCString, withCString)
import           Foreign.ForeignPtr          (withForeignPtr)
import           Foreign.Marshal.Alloc       (free)
import           Foreign.Ptr                 (nullPtr)
import           ISL.Core                    (Env (..), ISL, askEnv, manage,
                                              throwISL)
import           ISL.Internal.FFI
import           ISL.Internal.Schedule.Types (Schedule (..))
import           ISL.Internal.Set.Types      (UnionSet (..))

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

-- | Check if two schedules are equal (plain equality)
scheduleIsEqual :: Schedule s -> Schedule s -> ISL s Bool
scheduleIsEqual (Schedule fa) (Schedule fb) = do
    result <- liftIO $ withForeignPtr fa $ \pa ->
        withForeignPtr fb $ \pb -> c_sched_plain_is_equal pa pb
    case result of
        -1 -> throwISL "isl_schedule_plain_is_equal"
        0 -> pure False
        _ -> pure True
