{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.NVIDIA.Internal.Device (
    computeCapability,
) where

import           Control.Monad.IO.Class             (liftIO)
import           Foreign.C.Types                    (CInt (..))
import           Foreign.Marshal.Alloc              (alloca)
import           Foreign.Storable                   (peek)
import           Runtime.NVIDIA.Internal.Core       (Env (..), NVIDIA, askEnv,
                                                     expectDriverSuccess)
import           Runtime.NVIDIA.Internal.Driver.FFI

computeCapability :: NVIDIA s (Int, Int)
computeCapability = do
    env <- askEnv
    major <- queryDeviceAttribute cuDeviceAttributeComputeCapabilityMajor env.cudaDevice
    minor <- queryDeviceAttribute cuDeviceAttributeComputeCapabilityMinor env.cudaDevice
    pure (fromIntegral major, fromIntegral minor)

queryDeviceAttribute :: CInt -> CuDevice -> NVIDIA s CInt
queryDeviceAttribute attribute device = do
    (result, value) <-
        liftIO $
            alloca $ \valuePtr -> do
                result <- c_cuDeviceGetAttribute valuePtr attribute device
                value <- peek valuePtr
                pure (result, value)
    expectDriverSuccess "cuDeviceGetAttribute" result
    pure value
