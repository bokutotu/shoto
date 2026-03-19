{-# LANGUAGE CPP                 #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.NVIDIA (
    NVIDIA,
    CompiledCudaProgram,
    LoadedNvidiaKernel,
    DeviceBuffer,
    runNVIDIA,
    compileCudaProgram,
    loadNvidiaKernel,
    allocateDeviceBuffer,
    freeDeviceBuffer,
    uploadTensorBuffer,
    downloadTensorBuffer,
    runNvidiaKernel,
    runNvidiaKernelWithHostBuffers,
) where

import           Codegen.CUDA.Ast               (CudaDim (..))
import           Runtime.NVIDIA.Types           (CompiledCudaProgram,
                                                 DeviceBuffer,
                                                 LoadedNvidiaKernel, NVIDIA,
                                                 runNVIDIA)

#ifdef CUDA_RUNTIME
import           Control.Monad                  (unless, zipWithM_)
import           Control.Monad.Except           (MonadError (catchError, throwError))
import           Data.Foldable                  (traverse_)
import           Foreign.C.Types                (CFloat, CInt)
import           Foreign.Storable               (sizeOf)
import qualified Runtime.NVIDIA.Internal.Device as CUDA
import qualified Runtime.NVIDIA.Internal.Memory as CUDA
import qualified Runtime.NVIDIA.Internal.Module as CUDA
import qualified Runtime.NVIDIA.Internal.NVRTC  as CUDA
import           Runtime.NVIDIA.Types           (CompiledCudaProgram (..),
                                                 DeviceBuffer (..),
                                                 LoadedNvidiaKernel (..),
                                                 liftCuda)
import           Runtime.Types                  (KernelArg (..),
                                                 KernelSignature (..),
                                                 RuntimeError (..),
                                                 TensorBuffer (..))
#else
import           Control.Monad.Except           (throwError)
import           Runtime.Types                  (KernelArg, KernelSignature,
                                                 RuntimeError (..),
                                                 TensorBuffer)
#endif

compileCudaProgram :: KernelSignature -> CudaDim -> String -> NVIDIA s CompiledCudaProgram
#ifdef CUDA_RUNTIME
compileCudaProgram kernelSignature compiledCudaDim source = do
    (major, minor) <- liftCuda CUDA.computeCapability
    let compiledPtxOptions =
            [ "--gpu-architecture=compute_" <> show major <> show minor
            , "--std=c++11"
            ]
    compiledPtx <- liftCuda $ CUDA.compileProgramToPtx "shoto-runtime.cu" source compiledPtxOptions
    pure CompiledCudaProgram{compiledPtx, compiledKernelSignature = kernelSignature, compiledCudaDim}
#else
compileCudaProgram _ _ _ =
    throwError $ ErrRuntimeCudaUnavailable "compileCudaProgram"
#endif

loadNvidiaKernel :: CompiledCudaProgram -> NVIDIA s (LoadedNvidiaKernel s)
#ifdef CUDA_RUNTIME
loadNvidiaKernel compiledCudaProgram = do
    loadedModule <- liftCuda $ CUDA.loadModuleData compiledCudaProgram.compiledPtx
    loadedFunction <- liftCuda $ CUDA.getFunction loadedModule "shoto_kernel_cuda"
    pure
        LoadedNvidiaKernel
            { loadedKernelSignature = compiledCudaProgram.compiledKernelSignature
            , loadedCudaDim = compiledCudaProgram.compiledCudaDim
            , loadedModule
            , loadedFunction
            }
#else
loadNvidiaKernel _ =
    throwError $ ErrRuntimeCudaUnavailable "loadNvidiaKernel"
#endif

allocateDeviceBuffer :: Int -> NVIDIA s (DeviceBuffer s)
#ifdef CUDA_RUNTIME
allocateDeviceBuffer deviceBufferElements
    | deviceBufferElements < 0 =
        throwError $ ErrRuntimeInvalidDeviceBufferElements deviceBufferElements
    | otherwise = do
        deviceBufferPtr <- liftCuda $ CUDA.allocBytes (allocationByteCount deviceBufferElements)
        pure DeviceBuffer{deviceBufferPtr, deviceBufferElements}
#else
allocateDeviceBuffer _ =
    throwError $ ErrRuntimeCudaUnavailable "allocateDeviceBuffer"
#endif

freeDeviceBuffer :: DeviceBuffer s -> NVIDIA s ()
#ifdef CUDA_RUNTIME
freeDeviceBuffer deviceBuffer =
    liftCuda $ CUDA.freeDevicePtr deviceBuffer.deviceBufferPtr
#else
freeDeviceBuffer _ =
    throwError $ ErrRuntimeCudaUnavailable "freeDeviceBuffer"
#endif

uploadTensorBuffer :: TensorBuffer -> DeviceBuffer s -> NVIDIA s ()
#ifdef CUDA_RUNTIME
uploadTensorBuffer tensorBuffer deviceBuffer
    | deviceBuffer.deviceBufferElements < tensorBuffer.tensorElements =
        throwError $
            ErrRuntimeDeviceBufferTooSmall
                deviceBuffer.deviceBufferElements
                tensorBuffer.tensorElements
    | otherwise =
        liftCuda $
            CUDA.copyBytesToDeviceFromForeignPtr
                tensorBuffer.tensorData
                (tensorByteCount tensorBuffer.tensorElements)
                deviceBuffer.deviceBufferPtr
#else
uploadTensorBuffer _ _ =
    throwError $ ErrRuntimeCudaUnavailable "uploadTensorBuffer"
#endif

downloadTensorBuffer :: DeviceBuffer s -> TensorBuffer -> NVIDIA s ()
#ifdef CUDA_RUNTIME
downloadTensorBuffer deviceBuffer tensorBuffer
    | tensorBuffer.tensorElements < deviceBuffer.deviceBufferElements =
        throwError $
            ErrRuntimeHostBufferTooSmall
                tensorBuffer.tensorElements
                deviceBuffer.deviceBufferElements
    | otherwise =
        liftCuda $
            CUDA.copyBytesFromDeviceToForeignPtr
                deviceBuffer.deviceBufferPtr
                (tensorByteCount deviceBuffer.deviceBufferElements)
                tensorBuffer.tensorData
#else
downloadTensorBuffer _ _ =
    throwError $ ErrRuntimeCudaUnavailable "downloadTensorBuffer"
#endif

runNvidiaKernel :: LoadedNvidiaKernel s -> Int -> Int -> [DeviceBuffer s] -> NVIDIA s ()
#ifdef CUDA_RUNTIME
runNvidiaKernel loadedNvidiaKernel threadBlockSize extentValue deviceBuffers = do
    validateLaunch loadedNvidiaKernel.loadedKernelSignature threadBlockSize extentValue deviceBuffers
    let gridDim =
            dim3ForAxis
                loadedNvidiaKernel.loadedCudaDim
                (fromIntegral $ max 1 $ ceilDiv extentValue threadBlockSize)
        blockDim =
            dim3ForAxis
                loadedNvidiaKernel.loadedCudaDim
                (fromIntegral threadBlockSize)
        kernelArgs =
            CUDA.KernelArgInt extentValue
                : (CUDA.KernelArgDevicePtr . (.deviceBufferPtr) <$> deviceBuffers)
    liftCuda $ CUDA.launchKernel loadedNvidiaKernel.loadedFunction gridDim blockDim kernelArgs
    liftCuda CUDA.synchronize
#else
runNvidiaKernel _ _ _ _ =
    throwError $ ErrRuntimeCudaUnavailable "runNvidiaKernel"
#endif

runNvidiaKernelWithHostBuffers :: LoadedNvidiaKernel s -> Int -> [KernelArg] -> NVIDIA s ()
#ifdef CUDA_RUNTIME
runNvidiaKernelWithHostBuffers loadedNvidiaKernel threadBlockSize kernelArgs = do
    extentValue <-
        either throwError pure $ validateHostKernelArgs loadedNvidiaKernel.loadedKernelSignature kernelArgs
    let tensorBuffers = extractTensorBuffers kernelArgs
    deviceBuffers <- allocateAndUploadAll tensorBuffers
    let freeAllBuffers = traverse_ freeDeviceBuffer deviceBuffers
        downloadAllBuffers = zipWithM_ downloadTensorBuffer deviceBuffers tensorBuffers
    (runNvidiaKernel loadedNvidiaKernel threadBlockSize extentValue deviceBuffers >> downloadAllBuffers)
        `catchError` (\runtimeError -> freeAllBuffers >> throwError runtimeError)
    freeAllBuffers
#else
runNvidiaKernelWithHostBuffers _ _ _ =
    throwError $ ErrRuntimeCudaUnavailable "runNvidiaKernelWithHostBuffers"
#endif

#ifdef CUDA_RUNTIME
allocateAndUploadAll :: [TensorBuffer] -> NVIDIA s [DeviceBuffer s]
allocateAndUploadAll tensorBuffers = go [] tensorBuffers
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

validateLaunch :: KernelSignature -> Int -> Int -> [DeviceBuffer s] -> NVIDIA s ()
validateLaunch kernelSignature threadBlockSize extentValue deviceBuffers = do
    unless (threadBlockSize > 0 && threadBlockSize <= 1024) $
        throwError $
            ErrRuntimeInvalidThreadBlockSize threadBlockSize

    whenNegativeExtent extentValue
    whenExtentOutOfRange extentValue

    let expectedTensorCount = length kernelSignature.tensorParamNames
        actualTensorCount = length deviceBuffers
    unless (actualTensorCount == expectedTensorCount) $
        throwError $
            ErrRuntimeDeviceArgCountMismatch expectedTensorCount actualTensorCount

    zipWithM_ validateDeviceBuffer [1 :: Int ..] deviceBuffers
  where
    validateDeviceBuffer _ deviceBuffer
        | deviceBuffer.deviceBufferElements < extentValue =
            throwError $
                ErrRuntimeDeviceBufferTooSmall
                    deviceBuffer.deviceBufferElements
                    extentValue
        | otherwise = pure ()

validateHostKernelArgs :: KernelSignature -> [KernelArg] -> Either RuntimeError Int
validateHostKernelArgs kernelSignature kernelArgs = do
    let expectedArgCount = 1 + length kernelSignature.tensorParamNames
        actualArgCount = length kernelArgs
    unlessEither (actualArgCount == expectedArgCount) $
        ErrRuntimeArgCountMismatch expectedArgCount actualArgCount

    extentValue <-
        case kernelArgs of
            KernelArgInt hostExtentValue : _ -> Right hostExtentValue
            _ -> Left ErrRuntimeExpectedExtentArg

    if extentValue < 0
        then Left $ ErrRuntimeNegativeExtent extentValue
        else Right ()

    if extentValue > fromIntegral (maxBound :: CInt)
        then Left $ ErrRuntimeExtentOutOfRange extentValue
        else Right ()

    zipWithM_Either (validateTensorArg extentValue) [1 :: Int ..] (drop 1 kernelArgs)
    pure extentValue

validateTensorArg :: Int -> Int -> KernelArg -> Either RuntimeError ()
validateTensorArg extentValue argIndex kernelArg =
    case kernelArg of
        KernelArgInt _ -> Left $ ErrRuntimeExpectedTensorArg argIndex
        KernelArgTensor tensorBuffer
            | tensorBuffer.tensorElements < extentValue ->
                Left $
                    ErrRuntimeTensorTooSmall
                        argIndex
                        extentValue
                        tensorBuffer.tensorElements
            | otherwise -> Right ()

extractTensorBuffers :: [KernelArg] -> [TensorBuffer]
extractTensorBuffers kernelArgs =
    [ tensorBuffer
    | KernelArgTensor tensorBuffer <- drop 1 kernelArgs
    ]

dim3ForAxis :: CudaDim -> Int -> CUDA.Dim3
dim3ForAxis cudaDim selectedExtent =
    case cudaDim of
        CudaX -> CUDA.Dim3{dimX = fromIntegral selectedExtent, dimY = 1, dimZ = 1}
        CudaY -> CUDA.Dim3{dimX = 1, dimY = fromIntegral selectedExtent, dimZ = 1}
        CudaZ -> CUDA.Dim3{dimX = 1, dimY = 1, dimZ = fromIntegral selectedExtent}

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

whenNegativeExtent :: Int -> NVIDIA s ()
whenNegativeExtent extentValue =
    unless (extentValue >= 0) $
        throwError $
            ErrRuntimeNegativeExtent extentValue

whenExtentOutOfRange :: Int -> NVIDIA s ()
whenExtentOutOfRange extentValue =
    unless (extentValue <= fromIntegral (maxBound :: CInt)) $
        throwError $
            ErrRuntimeExtentOutOfRange extentValue

unlessEither :: Bool -> e -> Either e ()
unlessEither condition err =
    if condition
        then Right ()
        else Left err

zipWithM_Either :: (a -> b -> Either e ()) -> [a] -> [b] -> Either e ()
zipWithM_Either fn leftValues rightValues =
    sequence_ $ zipWith fn leftValues rightValues
#endif
