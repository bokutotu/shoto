{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedRecordDot        #-}
{-# LANGUAGE RankNTypes                 #-}

module Runtime.NVIDIA.Internal.Core (
    NVIDIA (..),
    Env (..),
    runNVIDIA,
    throwNVIDIA,
    askEnv,
    expectDriverSuccess,
    driverErrorFromResult,
) where

import           Control.Monad                      (when)
import           Control.Monad.Except               (ExceptT, MonadError,
                                                     runExceptT, throwError)
import           Control.Monad.IO.Class             (MonadIO, liftIO)
import           Control.Monad.Reader               (ReaderT, ask, runReaderT)
import           Control.Monad.Trans                (lift)
import           Foreign.C.String                   (peekCString)
import           Foreign.Marshal.Alloc              (alloca)
import           Foreign.Ptr                        (nullPtr)
import           Foreign.Storable                   (peek)
import           Runtime.NVIDIA.Internal.Driver.FFI
import           Runtime.Types                      (RuntimeError (..))

data Env = Env
    { cudaContext :: RawContext
    , cudaDevice :: CuDevice
    }

newtype NVIDIA s a = NVIDIA {unNVIDIA :: ExceptT RuntimeError (ReaderT Env IO) a}
    deriving (Functor, Applicative, Monad, MonadIO, MonadError RuntimeError)

askEnv :: NVIDIA s Env
askEnv = NVIDIA $ lift ask

throwNVIDIA :: RuntimeError -> NVIDIA s a
throwNVIDIA = throwError

runNVIDIA :: (forall s. NVIDIA s a) -> IO (Either RuntimeError a)
runNVIDIA action = do
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
                    contextResult <-
                        alloca $ \contextPtr -> do
                            result <- c_cuCtxCreate contextPtr 0 device
                            context <- peek contextPtr
                            pure (result, context)
                    case contextResult of
                        (result, _)
                            | result /= cuSuccess ->
                                Left <$> driverErrorFromResult "cuCtxCreate_v2" result
                        (_, rawContext) -> do
                            runResult <-
                                runReaderT
                                    (runExceptT action.unNVIDIA)
                                    Env
                                        { cudaContext = rawContext
                                        , cudaDevice = device
                                        }
                            destroyResult <- c_cuCtxDestroy rawContext
                            case runResult of
                                Left err -> pure $ Left err
                                Right value
                                    | destroyResult == cuSuccess -> pure $ Right value
                                    | otherwise ->
                                        Left <$> driverErrorFromResult "cuCtxDestroy_v2" destroyResult

expectDriverSuccess :: String -> CuResult -> NVIDIA s ()
expectDriverSuccess fnName result =
    when (result /= cuSuccess) $
        liftIO (driverErrorFromResult fnName result) >>= throwNVIDIA

driverErrorFromResult :: String -> CuResult -> IO RuntimeError
driverErrorFromResult fnName result = do
    cudaName <- lookupDriverErrorString c_cuGetErrorName result
    cudaMessage <- lookupDriverErrorString c_cuGetErrorString result
    pure $
        ErrRuntimeCudaDriverError
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
