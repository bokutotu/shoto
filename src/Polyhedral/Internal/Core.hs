{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes                 #-}

module Polyhedral.Internal.Core (
    -- * Types
    ISL (..),
    Env (..),
    PolyhedralError (..),
    IslError (..),

    -- * Core Functions
    runISL,
    manage,
    throwISL,
    askEnv,
) where

import           Control.Monad.Except    (ExceptT, MonadError, runExceptT,
                                          throwError)
import           Control.Monad.IO.Class  (MonadIO, liftIO)
import           Control.Monad.Reader    (ReaderT, ask, runReaderT)
import           Control.Monad.Trans     (lift)
import           Foreign.C.String        (peekCString)
import qualified Foreign.Concurrent      as FC
import           Foreign.ForeignPtr      (ForeignPtr, newForeignPtr,
                                          touchForeignPtr, withForeignPtr)
import           Foreign.Ptr             (Ptr, nullPtr)
import           Polyhedral.Error
import           Polyhedral.Internal.FFI

newtype Env = Env (ForeignPtr IslCtx)

askEnv :: ISL s Env
askEnv = ISL $ lift ask

newtype ISL s a = ISL {unISL :: ExceptT PolyhedralError (ReaderT Env IO) a}
    deriving (Functor, Applicative, Monad, MonadIO, MonadError PolyhedralError)

throwISL :: String -> ISL s a
throwISL fnName = do
    Env ctxFP <- askEnv
    (msg, file, line) <- liftIO $ withForeignPtr ctxFP $ \ctx -> do
        msgC <- c_ctx_last_error_msg ctx
        fileC <- c_ctx_last_error_file ctx
        lineC <- c_ctx_last_error_line ctx

        msg <- if msgC == nullPtr then pure Nothing else Just <$> peekCString msgC
        file <- if fileC == nullPtr then pure Nothing else Just <$> peekCString fileC
        let line = if lineC < 0 then Nothing else Just (fromIntegral lineC)
        pure (msg, file, line)

    throwError $ InternalIslError $ IslError fnName msg file line

runISL :: (forall s. ISL s a) -> IO (Either PolyhedralError a)
runISL action = do
    rawCtx <- c_ctx_alloc
    if rawCtx == nullPtr
        then
            pure $
                Left $
                    InternalIslError $
                        IslError "isl_ctx_alloc" (Just "Failed to allocate context") Nothing Nothing
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
