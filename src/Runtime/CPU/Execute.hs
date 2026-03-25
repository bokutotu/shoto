{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.CPU.Execute (
    withLoadedCPUKernel,
    runCPUKernel,
) where

import           Builder.CPU.Types          (CompiledSharedObject (..))
import           Control.Exception          (IOException, bracket, catch)
import           Control.Monad              (when)
import qualified Data.Map.Strict            as Map
import           Foreign.C.Types            (CInt (..))
import           Foreign.Marshal.Array      (withArray)
import           Foreign.Marshal.Utils      (with)
import           Foreign.Ptr                (FunPtr, Ptr, castPtr)
import           Runtime.Types              (KernelArg (..),
                                             KernelSignature (..),
                                             KernelTensorParam (..),
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
    let extentCount = length kernelSignature.extentParamNames
        expectedArgCount = extentCount + length kernelSignature.tensorParams
        actualArgCount = length kernelArgs
    when (actualArgCount /= expectedArgCount) $
        Left $
            ErrRuntimeArgCountMismatch expectedArgCount actualArgCount

    let (extentArgs, tensorArgs) = splitAt extentCount kernelArgs
    extentValues <- traverse expectExtentArg extentArgs
    mapM_ validateExtentValue extentValues
    let extentMap =
            Map.fromList $
                zip kernelSignature.extentParamNames extentValues

    zipWithM_
        (\argIndex (tensorParam, tensorArg) -> validateTensorArg extentMap argIndex tensorParam tensorArg)
        [1 :: Int ..]
        (zip kernelSignature.tensorParams tensorArgs)
  where
    expectExtentArg = \case
        KernelArgInt extentValue -> Right extentValue
        _ -> Left ErrRuntimeExpectedExtentArg

    validateExtentValue extentValue = do
        when (extentValue < 0) $
            Left $
                ErrRuntimeNegativeExtent extentValue
        when (extentValue > fromIntegral (maxBound :: CInt)) $
            Left $
                ErrRuntimeExtentOutOfRange extentValue

validateTensorArg ::
    Map.Map String Int ->
    Int ->
    KernelTensorParam ->
    KernelArg ->
    Either RuntimeError ()
validateTensorArg extentMap argIndex tensorParam = \case
    KernelArgInt _ -> Left $ ErrRuntimeExpectedTensorArg argIndex
    KernelArgTensor tensorBuffer
        | tensorBuffer.tensorElements < requiredElements ->
            Left $
                ErrRuntimeTensorTooSmall
                    argIndex
                    requiredElements
                    tensorBuffer.tensorElements
        | otherwise -> Right ()
      where
        requiredElements =
            product $
                fmap
                    (\shapeParamName -> Map.findWithDefault 1 shapeParamName extentMap)
                    tensorParam.tensorShapeParamNames

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
