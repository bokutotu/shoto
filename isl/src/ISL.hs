{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ISL (
    -- * Execution Scope
    ISL,
    runISL,
    liftIO,
    IslError (..),

    -- * Types (Opaque & Safe)
    SSet,
    SUnionSet,
    SSchedule,

    -- * Set Operations
    set,
    setToString,
    setUnion,
    (\/),
    setIntersect,
    (/\),
    setSubtract,
    (\\),
    setCoalesce,

    -- * Union Set Operations
    unionSet,
    unionSetToString,
    unionSetUnion,
    unionSetIntersect,
    unionSetSubtract,
    unionSetCoalesce,

    -- * Schedule Operations
    schedule,
    scheduleToString,
    scheduleDomain,
    scheduleFromDomain,
) where

import Control.Exception (Exception, bracket, mask_, throwIO)
import Control.Monad (when)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Reader (ReaderT, ask, runReaderT)
import Data.String (IsString (..))
import Foreign.C.String (CString, peekCString, withCString)
import Foreign.C.Types (CInt (..))
import Foreign.Concurrent qualified as FC
import Foreign.ForeignPtr (
    ForeignPtr,
    newForeignPtr,
    touchForeignPtr,
    withForeignPtr,
 )
import Foreign.Marshal.Alloc (free)
import Foreign.Ptr (FunPtr, Ptr, nullPtr)

data IslCtx

data IslSet

data IslUnionSet

data IslSchedule

type Ctx = Ptr IslCtx

type Set = Ptr IslSet

type UnionSet = Ptr IslUnionSet

type Schedule = Ptr IslSchedule

-- Context
foreign import ccall "isl/ctx.h isl_ctx_alloc"
    c_ctx_alloc :: IO Ctx

foreign import ccall "isl/ctx.h &isl_ctx_free"
    p_ctx_free :: FunPtr (Ctx -> IO ())

foreign import ccall "isl/ctx.h isl_ctx_last_error_msg"
    c_ctx_last_error_msg :: Ctx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_file"
    c_ctx_last_error_file :: Ctx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_line"
    c_ctx_last_error_line :: Ctx -> IO CInt

-- Set
foreign import ccall "isl/set.h isl_set_read_from_str"
    c_set_read :: Ctx -> CString -> IO Set

foreign import ccall "isl/set.h isl_set_to_str"
    c_set_to_str :: Set -> IO CString

foreign import ccall "isl/set.h isl_set_free"
    c_set_free :: Set -> IO ()

foreign import ccall "isl/set.h isl_set_copy"
    c_set_copy :: Set -> IO Set

foreign import ccall "isl/set.h isl_set_union"
    c_set_union :: Set -> Set -> IO Set

foreign import ccall "isl/set.h isl_set_intersect"
    c_set_intersect :: Set -> Set -> IO Set

foreign import ccall "isl/set.h isl_set_subtract"
    c_set_subtract :: Set -> Set -> IO Set

foreign import ccall "isl/set.h isl_set_coalesce"
    c_set_coalesce :: Set -> IO Set

-- Union Set
foreign import ccall "isl/union_set.h isl_union_set_read_from_str"
    c_uset_read :: Ctx -> CString -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_to_str"
    c_uset_to_str :: UnionSet -> IO CString

foreign import ccall "isl/union_set.h isl_union_set_free"
    c_uset_free :: UnionSet -> IO ()

foreign import ccall "isl/union_set.h isl_union_set_copy"
    c_uset_copy :: UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_union"
    c_uset_union :: UnionSet -> UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_intersect"
    c_uset_intersect :: UnionSet -> UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_subtract"
    c_uset_subtract :: UnionSet -> UnionSet -> IO UnionSet

foreign import ccall "isl/union_set.h isl_union_set_coalesce"
    c_uset_coalesce :: UnionSet -> IO UnionSet

-- Schedule
foreign import ccall "isl/schedule.h isl_schedule_read_from_str"
    c_sched_read :: Ctx -> CString -> IO Schedule

foreign import ccall "isl/schedule.h isl_schedule_to_str"
    c_sched_to_str :: Schedule -> IO CString

foreign import ccall "isl/schedule.h isl_schedule_free"
    c_sched_free :: Schedule -> IO ()

foreign import ccall "isl/schedule.h isl_schedule_from_domain"
    c_sched_from_domain :: UnionSet -> IO Schedule

foreign import ccall "isl/schedule.h isl_schedule_get_domain"
    c_sched_get_domain :: Schedule -> IO UnionSet

-- =========================================================
-- 2. Infrastructure (Monad & Safety)
-- =========================================================

-- | ISL実行環境。ContextへのForeignPtrを保持する。
newtype Env = Env (ForeignPtr IslCtx)

{- | ISLモナド。
's' はファントム型で、生成されたオブジェクトが runISL のスコープ外に出るのを防ぐ。
-}
newtype ISL s a = ISL {unISL :: ReaderT Env IO a}
    deriving (Functor, Applicative, Monad, MonadIO)

-- | 安全なオブジェクトラッパー
newtype SSet s = SSet (ForeignPtr IslSet)

newtype SUnionSet s = SUnionSet (ForeignPtr IslUnionSet)

newtype SSchedule s = SSchedule (ForeignPtr IslSchedule)

-- | ISLエラー情報
data IslError = IslError
    { islFunction :: String
    , islMessage :: Maybe String
    , islFile :: Maybe String
    , islLine :: Maybe Int
    }
    deriving (Show)

instance Exception IslError

-- | Ctxから最後のエラー詳細を取得して例外を投げる
throwNull :: String -> Ctx -> IO a
throwNull fnName ctx = do
    msgC <- c_ctx_last_error_msg ctx
    fileC <- c_ctx_last_error_file ctx
    lineC <- c_ctx_last_error_line ctx

    msg <- if msgC == nullPtr then pure Nothing else Just <$> peekCString msgC
    file <- if fileC == nullPtr then pure Nothing else Just <$> peekCString fileC
    let line = if lineC < 0 then Nothing else Just (fromIntegral lineC)

    throwIO $ IslError fnName msg file line

{- | ISL 計算の実行エントリポイント。
ランク2多相 (forall s.) により、内部リソースの外部流出を型レベルで阻止する。
-}
runISL :: (forall s. ISL s a) -> IO a
runISL action = do
    -- mask_ で非同期例外によるリソースリークを防ぐ
    ctxFP <- mask_ $ do
        rawCtx <- c_ctx_alloc
        when (rawCtx == nullPtr) $
            throwIO (IslError "isl_ctx_alloc" (Just "Failed to allocate context") Nothing Nothing)
        -- 標準のFinalizer (C関数) を使用
        newForeignPtr p_ctx_free rawCtx

    runReaderT (unISL action) (Env ctxFP)

{- | オブジェクト管理ヘルパー
1. 生成アクションを実行
2. エラーなら例外送出
3. 成功ならForeignPtr化し、Ctxへの依存関係(touch)を設定
-}
manage ::
    -- | 解放関数 (Raw)
    (Ptr a -> IO ()) ->
    -- | 関数名 (エラー用)
    String ->
    -- | 生成アクション
    IO (Ptr a) ->
    -- | コンストラクタ
    (ForeignPtr a -> b) ->
    ISL s b
manage rawFree fn producer ctor = ISL $ do
    Env ctxFP <- ask
    liftIO $ do
        ptr <- producer
        if ptr == nullPtr
            then withForeignPtr ctxFP $ \ctx -> throwNull fn ctx
            else do
                -- 依存関係付きファイナライザ (Haskell IO Action)
                fp <- FC.newForeignPtr ptr $ do
                    rawFree ptr
                    -- 【重要】子(Object)が解放される瞬間に親(Ctx)をtouchする
                    touchForeignPtr ctxFP
                return (ctor fp)

-- =========================================================
-- 3. Set Implementation
-- =========================================================

-- | 文字列リテラル対応: s <- "{ ... }"
instance IsString (ISL s (SSet s)) where
    fromString = set

set :: String -> ISL s (SSet s)
set str = ISL $ do
    Env ctxFP <- ask
    -- 生成アクションの定義
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_set_read ctx cstr
    unISL $ manage c_set_free "isl_set_read_from_str" mk SSet

setToString :: SSet s -> ISL s String
setToString (SSet fp) = ISL $ do
    Env ctxFP <- ask
    liftIO $ withForeignPtr fp $ \ptr -> do
        cstr <- c_set_to_str ptr
        if cstr == nullPtr
            then withForeignPtr ctxFP $ \ctx -> throwNull "isl_set_to_str" ctx
            else bracket (pure cstr) free peekCString

{- | 2項演算ヘルパー (Copy-on-Write)
入力をコピーしてから渡すことで、Haskell側の値を不変に保つ
-}
liftOp2Set ::
    (Set -> Set -> IO Set) ->
    String ->
    SSet s ->
    SSet s ->
    ISL s (SSet s)
liftOp2Set op name (SSet fa) (SSet fb) = do
    -- ForeignPtrは純粋に保持できるため、アクション内でtouchする
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_set_copy pa
                cb <- c_set_copy pb
                op ca cb
    manage c_set_free name mk SSet

setUnion :: SSet s -> SSet s -> ISL s (SSet s)
setUnion = liftOp2Set c_set_union "isl_set_union"

setIntersect :: SSet s -> SSet s -> ISL s (SSet s)
setIntersect = liftOp2Set c_set_intersect "isl_set_intersect"

setSubtract :: SSet s -> SSet s -> ISL s (SSet s)
setSubtract = liftOp2Set c_set_subtract "isl_set_subtract"

setCoalesce :: SSet s -> ISL s (SSet s)
setCoalesce (SSet fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_set_copy ptr
            c_set_coalesce cptr
    manage c_set_free "isl_set_coalesce" mk SSet

-- Operators
infixl 6 \/

infixl 7 /\

infixl 6 \\

(\/) :: SSet s -> SSet s -> ISL s (SSet s)
(\/) = setUnion

(/\) :: SSet s -> SSet s -> ISL s (SSet s)
(/\) = setIntersect

(\\) :: SSet s -> SSet s -> ISL s (SSet s)
(\\) = setSubtract

-- =========================================================
-- 4. Union Set Implementation
-- =========================================================

instance IsString (ISL s (SUnionSet s)) where
    fromString = unionSet

unionSet :: String -> ISL s (SUnionSet s)
unionSet str = ISL $ do
    Env ctxFP <- ask
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_uset_read ctx cstr
    unISL $ manage c_uset_free "isl_union_set_read_from_str" mk SUnionSet

unionSetToString :: SUnionSet s -> ISL s String
unionSetToString (SUnionSet fp) = ISL $ do
    Env ctxFP <- ask
    liftIO $ withForeignPtr fp $ \ptr -> do
        cstr <- c_uset_to_str ptr
        if cstr == nullPtr
            then withForeignPtr ctxFP $ \ctx -> throwNull "isl_union_set_to_str" ctx
            else bracket (pure cstr) free peekCString

liftOp2US ::
    (UnionSet -> UnionSet -> IO UnionSet) ->
    String ->
    SUnionSet s ->
    SUnionSet s ->
    ISL s (SUnionSet s)
liftOp2US op name (SUnionSet fa) (SUnionSet fb) = do
    let mk = withForeignPtr fa $ \pa ->
            withForeignPtr fb $ \pb -> do
                ca <- c_uset_copy pa
                cb <- c_uset_copy pb
                op ca cb
    manage c_uset_free name mk SUnionSet

unionSetUnion :: SUnionSet s -> SUnionSet s -> ISL s (SUnionSet s)
unionSetUnion = liftOp2US c_uset_union "isl_union_set_union"

unionSetIntersect :: SUnionSet s -> SUnionSet s -> ISL s (SUnionSet s)
unionSetIntersect = liftOp2US c_uset_intersect "isl_union_set_intersect"

unionSetSubtract :: SUnionSet s -> SUnionSet s -> ISL s (SUnionSet s)
unionSetSubtract = liftOp2US c_uset_subtract "isl_union_set_subtract"

unionSetCoalesce :: SUnionSet s -> ISL s (SUnionSet s)
unionSetCoalesce (SUnionSet fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_uset_copy ptr
            c_uset_coalesce cptr
    manage c_uset_free "isl_union_set_coalesce" mk SUnionSet

-- =========================================================
-- 5. Schedule Implementation
-- =========================================================

instance IsString (ISL s (SSchedule s)) where
    fromString = schedule

schedule :: String -> ISL s (SSchedule s)
schedule str = ISL $ do
    Env ctxFP <- ask
    let mk = withForeignPtr ctxFP $ \ctx ->
            withCString str $ \cstr -> c_sched_read ctx cstr
    unISL $ manage c_sched_free "isl_schedule_read_from_str" mk SSchedule

scheduleToString :: SSchedule s -> ISL s String
scheduleToString (SSchedule fp) = ISL $ do
    Env ctxFP <- ask
    liftIO $ withForeignPtr fp $ \ptr -> do
        cstr <- c_sched_to_str ptr
        if cstr == nullPtr
            then withForeignPtr ctxFP $ \ctx -> throwNull "isl_schedule_to_str" ctx
            else bracket (pure cstr) free peekCString

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
