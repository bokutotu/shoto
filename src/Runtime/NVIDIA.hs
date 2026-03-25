{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.NVIDIA (
    NVIDIA,
    LoadedNvidiaKernel,
    DeviceBuffer,
    runNVIDIA,
    loadNvidiaKernel,
    allocateDeviceBuffer,
    freeDeviceBuffer,
    uploadTensorBuffer,
    downloadTensorBuffer,
    runNvidiaKernel,
    runNvidiaKernelWithHostBuffers,
) where

import           Builder.NVIDIA.Types           (CompiledCudaProgram (..))
import           Control.Monad                  (unless, zipWithM_)
import           Control.Monad.Except           (MonadError (catchError, throwError))
import           Data.Foldable                  (traverse_)
import qualified Data.Map.Strict                as Map
import           Foreign.C.Types                (CFloat, CInt)
import           Foreign.Storable               (sizeOf)
import qualified Runtime.NVIDIA.Internal.Memory as CUDA
import qualified Runtime.NVIDIA.Internal.Module as CUDA
import           Runtime.NVIDIA.Types           (DeviceBuffer (..),
                                                 LoadedNvidiaKernel (..),
                                                 NVIDIA, runNVIDIA)
import           Runtime.Types                  (KernelArg (..),
                                                 KernelSignature (..),
                                                 KernelTensorParam (..),
                                                 RuntimeError (..),
                                                 TensorBuffer (..),
                                                 ThreadBlockShape (..))

loadNvidiaKernel :: CompiledCudaProgram -> NVIDIA s (LoadedNvidiaKernel s)
loadNvidiaKernel compiledCudaProgram = do
    loadedModule <- CUDA.loadModuleData compiledCudaProgram.compiledPtx
    loadedFunction <- CUDA.getFunction loadedModule "shoto_kernel_cuda"
    pure
        LoadedNvidiaKernel
            { loadedKernelSignature = compiledCudaProgram.compiledKernelSignature
            , loadedModule
            , loadedFunction
            }

allocateDeviceBuffer :: Int -> NVIDIA s (DeviceBuffer s)
allocateDeviceBuffer deviceBufferElements
    | deviceBufferElements < 0 =
        throwError $ ErrRuntimeInvalidDeviceBufferElements deviceBufferElements
    | otherwise = do
        deviceBufferPtr <- CUDA.allocBytes (allocationByteCount deviceBufferElements)
        pure DeviceBuffer{deviceBufferPtr, deviceBufferElements}

freeDeviceBuffer :: DeviceBuffer s -> NVIDIA s ()
freeDeviceBuffer deviceBuffer =
    CUDA.freeDevicePtr deviceBuffer.deviceBufferPtr

uploadTensorBuffer :: TensorBuffer -> DeviceBuffer s -> NVIDIA s ()
uploadTensorBuffer tensorBuffer deviceBuffer
    | deviceBuffer.deviceBufferElements < tensorBuffer.tensorElements =
        throwError $
            ErrRuntimeDeviceBufferTooSmall
                deviceBuffer.deviceBufferElements
                tensorBuffer.tensorElements
    | otherwise =
        CUDA.copyBytesToDeviceFromForeignPtr
            tensorBuffer.tensorData
            (tensorByteCount tensorBuffer.tensorElements)
            deviceBuffer.deviceBufferPtr

downloadTensorBuffer :: DeviceBuffer s -> TensorBuffer -> NVIDIA s ()
downloadTensorBuffer deviceBuffer tensorBuffer
    | tensorBuffer.tensorElements < deviceBuffer.deviceBufferElements =
        throwError $
            ErrRuntimeHostBufferTooSmall
                tensorBuffer.tensorElements
                deviceBuffer.deviceBufferElements
    | otherwise =
        CUDA.copyBytesFromDeviceToForeignPtr
            deviceBuffer.deviceBufferPtr
            (tensorByteCount deviceBuffer.deviceBufferElements)
            tensorBuffer.tensorData

runNvidiaKernel ::
    LoadedNvidiaKernel s ->
    ThreadBlockShape ->
    [Int] ->
    [DeviceBuffer s] ->
    NVIDIA s ()
runNvidiaKernel loadedNvidiaKernel threadBlockShape extentValues deviceBuffers = do
    validateLaunch loadedNvidiaKernel.loadedKernelSignature threadBlockShape extentValues deviceBuffers
    let blockDim = blockDimFromShape threadBlockShape
    gridDim <- either throwError pure $ gridDimForExtents threadBlockShape extentValues
    let kernelArgs =
            (CUDA.KernelArgInt <$> extentValues)
                <> (CUDA.KernelArgDevicePtr . (.deviceBufferPtr) <$> deviceBuffers)
    CUDA.launchKernel loadedNvidiaKernel.loadedFunction gridDim blockDim kernelArgs
    CUDA.synchronize

runNvidiaKernelWithHostBuffers ::
    LoadedNvidiaKernel s ->
    ThreadBlockShape ->
    [KernelArg] ->
    NVIDIA s ()
runNvidiaKernelWithHostBuffers loadedNvidiaKernel threadBlockShape kernelArgs = do
    (extentValues, tensorBuffers) <-
        either throwError pure $
            validateHostKernelArgs loadedNvidiaKernel.loadedKernelSignature kernelArgs
    deviceBuffers <- allocateAndUploadAll tensorBuffers
    let freeAllBuffers = traverse_ freeDeviceBuffer deviceBuffers
        downloadAllBuffers = zipWithM_ downloadTensorBuffer deviceBuffers tensorBuffers
    (runNvidiaKernel loadedNvidiaKernel threadBlockShape extentValues deviceBuffers >> downloadAllBuffers)
        `catchError` (\runtimeError -> freeAllBuffers >> throwError runtimeError)
    freeAllBuffers

allocateAndUploadAll :: [TensorBuffer] -> NVIDIA s [DeviceBuffer s]
allocateAndUploadAll = go []
  where
    go reversedDeviceBuffers [] =
        pure $ reverse reversedDeviceBuffers
    go reversedDeviceBuffers (tensorBuffer : remainingTensorBuffers) =
        ( do
            deviceBuffer <- allocateDeviceBuffer tensorBuffer.tensorElements
            uploadTensorBuffer tensorBuffer deviceBuffer
                `catchError` \runtimeError -> do
                    freeDeviceBuffer deviceBuffer
                    throwError runtimeError
            go (deviceBuffer : reversedDeviceBuffers) remainingTensorBuffers
        )
            `catchError` \runtimeError -> do
                traverse_ freeDeviceBuffer reversedDeviceBuffers
                throwError runtimeError

validateLaunch ::
    KernelSignature ->
    ThreadBlockShape ->
    [Int] ->
    [DeviceBuffer s] ->
    NVIDIA s ()
validateLaunch kernelSignature threadBlockShape extentValues deviceBuffers = do
    either throwError pure $ validateThreadBlockShape threadBlockShape
    either throwError pure $ validateExtentValues extentValues

    let expectedExtentCount = length kernelSignature.extentParamNames
        actualExtentCount = length extentValues
    unless (actualExtentCount == expectedExtentCount) $
        throwError $
            ErrRuntimeArgCountMismatch expectedExtentCount actualExtentCount

    let expectedTensorCount = length kernelSignature.tensorParams
        actualTensorCount = length deviceBuffers
    unless (actualTensorCount == expectedTensorCount) $
        throwError $
            ErrRuntimeDeviceArgCountMismatch expectedTensorCount actualTensorCount

    let extentMap = extentValueMap kernelSignature extentValues
    zipWithM_
        (validateDeviceBuffer extentMap)
        [1 :: Int ..]
        (zip kernelSignature.tensorParams deviceBuffers)

validateDeviceBuffer ::
    Map.Map String Int ->
    Int ->
    (KernelTensorParam, DeviceBuffer s) ->
    NVIDIA s ()
validateDeviceBuffer extentMap _argIndex (tensorParam, deviceBuffer)
    | deviceBuffer.deviceBufferElements < requiredElements =
        throwError $
            ErrRuntimeDeviceBufferTooSmall
                deviceBuffer.deviceBufferElements
                requiredElements
    | otherwise = pure ()
  where
    requiredElements = requiredTensorElements extentMap tensorParam

validateHostKernelArgs ::
    KernelSignature ->
    [KernelArg] ->
    Either RuntimeError ([Int], [TensorBuffer])
validateHostKernelArgs kernelSignature kernelArgs = do
    let extentCount = length kernelSignature.extentParamNames
        expectedArgCount = extentCount + length kernelSignature.tensorParams
        actualArgCount = length kernelArgs
    unlessEither (actualArgCount == expectedArgCount) $
        ErrRuntimeArgCountMismatch expectedArgCount actualArgCount

    let (extentArgs, tensorArgs) = splitAt extentCount kernelArgs
    extentValues <- traverse expectExtentArg extentArgs
    mapM_ validateExtentValue extentValues
    let extentMap = extentValueMap kernelSignature extentValues
    tensorBuffers <-
        zipWithMEither
            (\argIndex (tensorParam, kernelArg) -> validateTensorArg extentMap argIndex tensorParam kernelArg)
            [1 :: Int ..]
            (zip kernelSignature.tensorParams tensorArgs)
    pure (extentValues, tensorBuffers)
  where
    expectExtentArg = \case
        KernelArgInt extentValue -> Right extentValue
        _ -> Left ErrRuntimeExpectedExtentArg

validateTensorArg ::
    Map.Map String Int ->
    Int ->
    KernelTensorParam ->
    KernelArg ->
    Either RuntimeError TensorBuffer
validateTensorArg extentMap argIndex tensorParam kernelArg =
    case kernelArg of
        KernelArgInt _ -> Left $ ErrRuntimeExpectedTensorArg argIndex
        KernelArgTensor tensorBuffer
            | tensorBuffer.tensorElements < requiredElements ->
                Left $
                    ErrRuntimeTensorTooSmall
                        argIndex
                        requiredElements
                        tensorBuffer.tensorElements
            | otherwise -> Right tensorBuffer
  where
    requiredElements = requiredTensorElements extentMap tensorParam

requiredTensorElements :: Map.Map String Int -> KernelTensorParam -> Int
requiredTensorElements extentMap tensorParam =
    product $
        fmap
            (\shapeParamName -> Map.findWithDefault 1 shapeParamName extentMap)
            tensorParam.tensorShapeParamNames

extentValueMap :: KernelSignature -> [Int] -> Map.Map String Int
extentValueMap kernelSignature extentValues =
    Map.fromList $
        zip kernelSignature.extentParamNames extentValues

validateThreadBlockShape :: ThreadBlockShape -> Either RuntimeError ()
validateThreadBlockShape threadBlockShape = do
    mapM_
        validatePositive
        [threadBlockShape.blockDimX, threadBlockShape.blockDimY, threadBlockShape.blockDimZ]
    unlessEither (threadCount <= 1024) $
        ErrRuntimeCudaUsageError "thread block shape exceeds 1024 threads"
  where
    threadCount =
        threadBlockShape.blockDimX
            * threadBlockShape.blockDimY
            * threadBlockShape.blockDimZ

    validatePositive dim =
        unlessEither (dim > 0) $
            ErrRuntimeInvalidThreadBlockSize dim

validateExtentValues :: [Int] -> Either RuntimeError ()
validateExtentValues =
    mapM_ validateExtentValue

validateExtentValue :: Int -> Either RuntimeError ()
validateExtentValue extentValue = do
    whenNegativeExtent extentValue
    whenExtentOutOfRange extentValue

blockDimFromShape :: ThreadBlockShape -> CUDA.Dim3
blockDimFromShape threadBlockShape =
    CUDA.Dim3
        { dimX = fromIntegral threadBlockShape.blockDimX
        , dimY = fromIntegral threadBlockShape.blockDimY
        , dimZ = fromIntegral threadBlockShape.blockDimZ
        }

gridDimForExtents :: ThreadBlockShape -> [Int] -> Either RuntimeError CUDA.Dim3
gridDimForExtents threadBlockShape extentValues =
    case reverse extentValues of
        [] ->
            Right CUDA.Dim3{dimX = 1, dimY = 1, dimZ = 1}
        [xExtent] ->
            Right
                CUDA.Dim3
                    { dimX = gridExtent xExtent threadBlockShape.blockDimX
                    , dimY = 1
                    , dimZ = 1
                    }
        [xExtent, yExtent] ->
            Right
                CUDA.Dim3
                    { dimX = gridExtent xExtent threadBlockShape.blockDimX
                    , dimY = gridExtent yExtent threadBlockShape.blockDimY
                    , dimZ = 1
                    }
        [xExtent, yExtent, zExtent] ->
            Right
                CUDA.Dim3
                    { dimX = gridExtent xExtent threadBlockShape.blockDimX
                    , dimY = gridExtent yExtent threadBlockShape.blockDimY
                    , dimZ = gridExtent zExtent threadBlockShape.blockDimZ
                    }
        _ ->
            Left $
                ErrRuntimeCudaUsageError
                    "CUDA runtime supports at most 3 extent parameters"
  where
    gridExtent extentValue blockDim =
        fromIntegral $
            max 1 $
                ceilDiv extentValue blockDim

ceilDiv :: Int -> Int -> Int
ceilDiv numerator denominator =
    if numerator <= 0
        then 1
        else (numerator + denominator - 1) `div` denominator

tensorByteCount :: Int -> Int
tensorByteCount tensorElements =
    tensorElements * sizeOf (undefined :: CFloat)

allocationByteCount :: Int -> Int
allocationByteCount tensorElements =
    max 1 (tensorByteCount tensorElements)

whenNegativeExtent :: Int -> Either RuntimeError ()
whenNegativeExtent extentValue =
    unlessEither (extentValue >= 0) $
        ErrRuntimeNegativeExtent extentValue

whenExtentOutOfRange :: Int -> Either RuntimeError ()
whenExtentOutOfRange extentValue =
    unlessEither (extentValue <= fromIntegral (maxBound :: CInt)) $
        ErrRuntimeExtentOutOfRange extentValue

unlessEither :: Bool -> e -> Either e ()
unlessEither condition err =
    if condition
        then Right ()
        else Left err

zipWithMEither :: (a -> b -> Either e c) -> [a] -> [b] -> Either e [c]
zipWithMEither func leftValues rightValues =
    sequence $ zipWith func leftValues rightValues
