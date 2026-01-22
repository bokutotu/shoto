{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes                 #-}

module ISL.Core (
    -- * Types
    ISL (..),
    Env (..),
    IslError (..),

    -- * Core Functions
    runISL,
    manage,
    throwISL,
    askEnv,
) where

import           Control.Monad.Except   (ExceptT, MonadError, runExceptT,
                                         throwError)
import           Control.Monad.IO.Class (MonadIO, liftIO)
import           Control.Monad.Reader   (ReaderT, ask, runReaderT)
import           Control.Monad.Trans    (lift)
import           Foreign.C.String       (peekCString)
import qualified Foreign.Concurrent     as FC
import           Foreign.ForeignPtr     (ForeignPtr, newForeignPtr,
                                         touchForeignPtr, withForeignPtr)
import           Foreign.Ptr            (Ptr, nullPtr)
import           ISL.Internal.FFI

newtype Env = Env (ForeignPtr IslCtx)

askEnv :: ISL s Env
askEnv = ISL $ lift ask

newtype ISL s a = ISL {unISL :: ExceptT IslError (ReaderT Env IO) a}
    deriving (Functor, Applicative, Monad, MonadIO, MonadError IslError)

data IslError = IslError
    { islFunction :: String
    , islMessage  :: Maybe String
    , islFile     :: Maybe String
    , islLine     :: Maybe Int
    }
    deriving (Show, Eq)

getCtx :: ISL s RawCtx
getCtx = do
    Env ctxFP <- askEnv
    liftIO $ withForeignPtr ctxFP pure

throwISL :: String -> ISL s a
throwISL fnName = do
    ctx <- getCtx
    msgC <- liftIO $ c_ctx_last_error_msg ctx
    fileC <- liftIO $ c_ctx_last_error_file ctx
    lineC <- liftIO $ c_ctx_last_error_line ctx

    msg <- liftIO $ if msgC == nullPtr then pure Nothing else Just <$> peekCString msgC
    file <- liftIO $ if fileC == nullPtr then pure Nothing else Just <$> peekCString fileC
    let line = if lineC < 0 then Nothing else Just (fromIntegral lineC)

    throwError $ IslError fnName msg file line

runISL :: (forall s. ISL s a) -> IO (Either IslError a)
runISL action = do
    rawCtx <- c_ctx_alloc
    if rawCtx == nullPtr
        then pure $ Left $ IslError "isl_ctx_alloc" (Just "Failed to allocate context") Nothing Nothing
        else do
            ctxFP <- newForeignPtr p_ctx_free rawCtx
            runReaderT (runExceptT (unISL action)) (Env ctxFP)

manage :: (Ptr a -> IO ()) -> String -> IO (Ptr a) -> (ForeignPtr a -> b) -> ISL s b
manage rawFree fn producer ctor = do
    Env ctxFP <- askEnv
    ptr <- liftIO producer
    if ptr == nullPtr
        then throwISL fn
        else liftIO $ do
            fp <- FC.newForeignPtr ptr $ do
                rawFree ptr
                touchForeignPtr ctxFP
            pure (ctor fp)
