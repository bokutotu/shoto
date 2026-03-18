{-# LANGUAGE OverloadedRecordDot #-}

module CUDA.Device (
    computeCapability,
) where

import CUDA.Core (
    CUDA,
    Env (..),
    askEnv,
    expectDriverSuccess,
 )
import CUDA.Internal.Driver.FFI
import Control.Monad.IO.Class (liftIO)
import Foreign.C.Types (CInt (..))
import Foreign.Marshal.Alloc (alloca)
import Foreign.Storable (peek)

computeCapability :: CUDA s (Int, Int)
computeCapability = do
    env <- askEnv
    major <- queryDeviceAttribute cuDeviceAttributeComputeCapabilityMajor env.cudaDevice
    minor <- queryDeviceAttribute cuDeviceAttributeComputeCapabilityMinor env.cudaDevice
    pure (fromIntegral major, fromIntegral minor)

queryDeviceAttribute :: CInt -> CuDevice -> CUDA s CInt
queryDeviceAttribute attribute device = do
    (result, value) <-
        liftIO $
            alloca $ \valuePtr -> do
                result <- c_cuDeviceGetAttribute valuePtr attribute device
                value <- peek valuePtr
                pure (result, value)
    expectDriverSuccess "cuDeviceGetAttribute" result
    pure value
