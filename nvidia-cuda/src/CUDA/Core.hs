{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedRecordDot        #-}
{-# LANGUAGE RankNTypes                 #-}

module CUDA.Core (
    CUDA (..),
    Env (..),
    CudaError (..),
    runCUDA,
    throwCUDA,
    askEnv,
    expectDriverSuccess,
    expectNvrtcSuccess,
    driverErrorFromResult,
    nvrtcErrorFromResult,
) where

import           Control.Monad            (when)
import           Control.Monad.Except     (ExceptT, MonadError, runExceptT,
                                           throwError)
import           Control.Monad.IO.Class   (MonadIO, liftIO)
import           Control.Monad.Reader     (ReaderT, ask, runReaderT)
import           Control.Monad.Trans      (lift)
import           CUDA.Internal.Driver.FFI
import           CUDA.Internal.NVRTC.FFI
import           Foreign.C.String         (peekCString)
import           Foreign.Marshal.Alloc    (alloca)
import           Foreign.Ptr              (nullPtr)
import           Foreign.Storable         (peek)

data Env = Env
    { cudaContext :: RawContext
    , cudaDevice :: CuDevice
    }

newtype CUDA s a = CUDA {unCUDA :: ExceptT CudaError (ReaderT Env IO) a}
    deriving (Functor, Applicative, Monad, MonadIO, MonadError CudaError)

data CudaError
    = CudaDriverError
        { cudaFunction :: String
        , cudaCode :: Int
        , cudaName :: Maybe String
        , cudaMessage :: Maybe String
        }
    | CudaNvrtcError
        { cudaFunction :: String
        , cudaCode :: Int
        , cudaMessage :: Maybe String
        , cudaLog :: Maybe String
        }
    | CudaUsageError String
    deriving (Eq, Show)

askEnv :: CUDA s Env
askEnv = CUDA $ lift ask

throwCUDA :: CudaError -> CUDA s a
throwCUDA = throwError

runCUDA :: (forall s. CUDA s a) -> IO (Either CudaError a)
runCUDA action = do
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
                                    (runExceptT (action.unCUDA))
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

expectDriverSuccess :: String -> CuResult -> CUDA s ()
expectDriverSuccess fnName result =
    when (result /= cuSuccess) $
        liftIO (driverErrorFromResult fnName result) >>= throwCUDA

expectNvrtcSuccess :: String -> NvrtcResult -> Maybe String -> CUDA s ()
expectNvrtcSuccess fnName result compileLog =
    when (result /= nvrtcSuccess) $
        liftIO (nvrtcErrorFromResult fnName result compileLog) >>= throwCUDA

driverErrorFromResult :: String -> CuResult -> IO CudaError
driverErrorFromResult fnName result = do
    cudaName <- lookupDriverErrorString c_cuGetErrorName result
    cudaMessage <- lookupDriverErrorString c_cuGetErrorString result
    pure
        CudaDriverError
            { cudaFunction = fnName
            , cudaCode = fromIntegral result
            , cudaName
            , cudaMessage
            }

nvrtcErrorFromResult :: String -> NvrtcResult -> Maybe String -> IO CudaError
nvrtcErrorFromResult fnName result compileLog = do
    messagePtr <- c_nvrtcGetErrorString result
    cudaMessage <-
        if messagePtr == nullPtr
            then pure Nothing
            else Just <$> peekCString messagePtr
    pure
        CudaNvrtcError
            { cudaFunction = fnName
            , cudaCode = fromIntegral result
            , cudaMessage
            , cudaLog = compileLog
            }

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
