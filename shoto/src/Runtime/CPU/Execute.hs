{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.CPU.Execute (
    withLoadedCPUKernel,
    runCPUKernel,
) where

import           Control.Exception          (IOException, bracket, catch)
import           Foreign.C.Types            (CInt (..))
import           Foreign.Marshal.Array      (withArray)
import           Foreign.Marshal.Utils      (with)
import           Foreign.Ptr                (FunPtr, Ptr, castPtr)
import           Runtime.CPU.JIT            (CompiledSharedObject (..))
import           Runtime.Types              (KernelArg (..),
                                             KernelSignature (..),
                                             RuntimeError (..),
                                             TensorBuffer (..),
                                             withTensorBufferPtr)
import           System.Posix.DynamicLinker (DL,
                                             RTLDFlags (RTLD_LOCAL, RTLD_NOW),
                                             dlclose, dlopen, dlsym)

data LoadedCPUKernel = LoadedCPUKernel
    { dynamicLibrary :: DL
    , loadedKernelSignature :: KernelSignature
    , dispatchFunction :: FunPtr DispatchFunction
    }

type DispatchFunction = CInt -> Ptr (Ptr ()) -> IO ()

foreign import ccall "dynamic" mkDispatchFunction :: FunPtr DispatchFunction -> DispatchFunction

withLoadedCPUKernel ::
    CompiledSharedObject -> (LoadedCPUKernel -> IO a) -> IO (Either RuntimeError a)
withLoadedCPUKernel compiledSharedObject action =
    loadKernel >>= \case
        Left err -> pure $ Left err
        Right loadedKernel ->
            Right <$> bracket (pure loadedKernel) unloadKernel action
  where
    loadKernel = do
        dlopenResult <-
            catchIOException
                (Right <$> dlopen compiledSharedObject.sharedObjectPath [RTLD_NOW, RTLD_LOCAL])
                (pure . Left . renderLoadError)
        case dlopenResult of
            Left err -> pure $ Left err
            Right dynamicLibrary -> do
                dispatchResult <-
                    catchIOException
                        (Right <$> dlsym dynamicLibrary "shoto_dispatch")
                        (pure . Left . renderSymbolError)
                case dispatchResult of
                    Left err -> do
                        ignoreIOException $ dlclose dynamicLibrary
                        pure $ Left err
                    Right dispatchFunction ->
                        pure $
                            Right
                                LoadedCPUKernel
                                    { dynamicLibrary
                                    , loadedKernelSignature = compiledSharedObject.kernelSignature
                                    , dispatchFunction
                                    }

    unloadKernel loadedKernel =
        ignoreIOException $ dlclose loadedKernel.dynamicLibrary

    renderLoadError loadException =
        ErrRuntimeLoadFailed compiledSharedObject.sharedObjectPath (show loadException)

    renderSymbolError symbolException =
        ErrRuntimeSymbolFailed compiledSharedObject.sharedObjectPath "shoto_dispatch" (show symbolException)

runCPUKernel :: LoadedCPUKernel -> [KernelArg] -> IO (Either RuntimeError ())
runCPUKernel loadedKernel kernelArgs =
    case validateKernelArgs loadedKernel.loadedKernelSignature kernelArgs of
        Left err -> pure $ Left err
        Right () ->
            withKernelArgPointers kernelArgs $ \argPointers ->
                withArray argPointers $ \argVector -> do
                    mkDispatchFunction
                        loadedKernel.dispatchFunction
                        (fromIntegral $ length argPointers)
                        argVector
                    pure $ Right ()

validateKernelArgs :: KernelSignature -> [KernelArg] -> Either RuntimeError ()
validateKernelArgs kernelSignature kernelArgs = do
    let expectedArgCount = 1 + length kernelSignature.tensorParamNames
        actualArgCount = length kernelArgs
    if actualArgCount /= expectedArgCount
        then Left $ ErrRuntimeArgCountMismatch expectedArgCount actualArgCount
        else pure ()

    extentValue <-
        case kernelArgs of
            KernelArgInt extentValue : _ -> Right extentValue
            _ -> Left ErrRuntimeExpectedExtentArg

    if extentValue < 0
        then Left $ ErrRuntimeNegativeExtent extentValue
        else pure ()

    if extentValue > fromIntegral (maxBound :: CInt)
        then Left $ ErrRuntimeExtentOutOfRange extentValue
        else pure ()

    let tensorArgs = drop 1 kernelArgs
    zipWithM_ (validateTensorArg extentValue) [1 ..] tensorArgs

validateTensorArg :: Int -> Int -> KernelArg -> Either RuntimeError ()
validateTensorArg extentValue argIndex = \case
    KernelArgInt _ -> Left $ ErrRuntimeExpectedTensorArg argIndex
    KernelArgTensor tensorBuffer
        | tensorBuffer.tensorElements < extentValue ->
            Left $
                ErrRuntimeTensorTooSmall
                    argIndex
                    extentValue
                    tensorBuffer.tensorElements
        | otherwise -> Right ()

withKernelArgPointers :: [KernelArg] -> ([Ptr ()] -> IO a) -> IO a
withKernelArgPointers kernelArgs continue =
    go kernelArgs []
  where
    go [] reversedPointers =
        continue $ reverse reversedPointers
    go (kernelArg : remainingArgs) reversedPointers =
        case kernelArg of
            KernelArgInt extentValue ->
                with (fromIntegral extentValue :: CInt) $ \extentPtr ->
                    go remainingArgs (castPtr extentPtr : reversedPointers)
            KernelArgTensor tensorBuffer ->
                withTensorBufferPtr tensorBuffer $ \tensorPtr ->
                    go remainingArgs (castPtr tensorPtr : reversedPointers)

ignoreIOException :: IO () -> IO ()
ignoreIOException action =
    catch action handler
  where
    handler :: IOException -> IO ()
    handler _ = pure ()

catchIOException :: IO a -> (IOException -> IO a) -> IO a
catchIOException = catch

zipWithM_ :: (a -> b -> Either e ()) -> [a] -> [b] -> Either e ()
zipWithM_ func leftValues rightValues =
    sequence_ $ zipWith func leftValues rightValues
