module Builder.NVIDIA.Internal.Device (
    computeCapability,
) where

import           Builder.Types                      (BuilderError (..))
import           Foreign.C.String                   (peekCString)
import           Foreign.C.Types                    (CInt)
import           Foreign.Marshal.Alloc              (alloca)
import           Foreign.Ptr                        (nullPtr)
import           Foreign.Storable                   (peek)
import           Runtime.NVIDIA.Internal.Driver.FFI

computeCapability :: IO (Either BuilderError (Int, Int))
computeCapability = do
    initResult <- c_cuInit 0
    if initResult /= cuSuccess
        then Left <$> driverErrorFromResult "cuInit" initResult
        else do
            deviceResult <-
                alloca $ \devicePtr -> do
                    result <- c_cuDeviceGet devicePtr 0
                    device <- peek devicePtr
                    pure (result, device)
            case deviceResult of
                (result, _)
                    | result /= cuSuccess ->
                        Left <$> driverErrorFromResult "cuDeviceGet" result
                (_, device) -> do
                    majorResult <- queryDeviceAttribute cuDeviceAttributeComputeCapabilityMajor device
                    minorResult <- queryDeviceAttribute cuDeviceAttributeComputeCapabilityMinor device
                    pure $ do
                        major <- majorResult
                        minor <- minorResult
                        Right (fromIntegral major, fromIntegral minor)

queryDeviceAttribute :: CInt -> CuDevice -> IO (Either BuilderError CInt)
queryDeviceAttribute attribute device = do
    (result, value) <-
        alloca $ \valuePtr -> do
            result <- c_cuDeviceGetAttribute valuePtr attribute device
            value <- peek valuePtr
            pure (result, value)
    if result /= cuSuccess
        then Left <$> driverErrorFromResult "cuDeviceGetAttribute" result
        else pure $ Right value

driverErrorFromResult :: String -> CuResult -> IO BuilderError
driverErrorFromResult fnName result = do
    cudaName <- lookupDriverErrorString c_cuGetErrorName result
    cudaMessage <- lookupDriverErrorString c_cuGetErrorString result
    pure $
        ErrBuilderCudaDriverError
            fnName
            (fromIntegral result)
            cudaName
            cudaMessage

lookupDriverErrorString ::
    (CuResult -> CStringResultPtr -> IO CuResult) ->
    CuResult ->
    IO (Maybe String)
lookupDriverErrorString lookupFn result =
    alloca $ \valuePtr -> do
        lookupResult <- lookupFn result valuePtr
        value <- peek valuePtr
        if lookupResult /= cuSuccess || value == nullPtr
            then pure Nothing
            else Just <$> peekCString value
