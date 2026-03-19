{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.Types (
    KernelSignature (..),
    TensorBuffer (..),
    KernelArg (..),
    RuntimeError (..),
    emptyTensorBuffer,
    tensorBufferFromList,
    readTensorBuffer,
    withTensorBufferPtr,
) where

import           Control.Monad         (when)
import           Foreign.C.Types       (CFloat)
import           Foreign.ForeignPtr    (ForeignPtr, mallocForeignPtrArray,
                                        withForeignPtr)
import           Foreign.Marshal.Array (peekArray, pokeArray)
import           Foreign.Ptr           (Ptr)
import           System.Exit           (ExitCode)

data KernelSignature = KernelSignature
    { extentParamName :: String
    , tensorParamNames :: [String]
    }
    deriving (Eq, Show)

data TensorBuffer = TensorBuffer
    { tensorData :: ForeignPtr CFloat
    , tensorElements :: Int
    }

instance Show TensorBuffer where
    show tensorBuffer =
        "TensorBuffer { tensorElements = " <> show tensorBuffer.tensorElements <> " }"

data KernelArg
    = KernelArgInt Int
    | KernelArgTensor TensorBuffer
    deriving (Show)

data RuntimeError
    = ErrRuntimeGccFailed FilePath ExitCode String String
    | ErrRuntimeLoadFailed FilePath String
    | ErrRuntimeSymbolFailed FilePath String String
    | ErrRuntimeCudaUnavailable String
    | ErrRuntimeArgCountMismatch Int Int
    | ErrRuntimeExpectedExtentArg
    | ErrRuntimeExpectedTensorArg Int
    | ErrRuntimeNegativeExtent Int
    | ErrRuntimeExtentOutOfRange Int
    | ErrRuntimeTensorTooSmall Int Int Int
    | ErrRuntimeDeviceArgCountMismatch Int Int
    | ErrRuntimeInvalidThreadBlockSize Int
    | ErrRuntimeInvalidDeviceBufferElements Int
    | ErrRuntimeDeviceBufferTooSmall Int Int
    | ErrRuntimeHostBufferTooSmall Int Int
    | ErrRuntimeCudaDriverError String Int (Maybe String) (Maybe String)
    | ErrRuntimeCudaNvrtcError String Int (Maybe String) (Maybe String)
    deriving (Eq, Show)

emptyTensorBuffer :: Int -> IO TensorBuffer
emptyTensorBuffer tensorElements = do
    when (tensorElements < 0) $
        ioError $
            userError "emptyTensorBuffer: negative element count"
    tensorData <- mallocForeignPtrArray tensorElements
    withForeignPtr tensorData $ \tensorPtr ->
        pokeArray tensorPtr (replicate tensorElements 0)
    pure TensorBuffer{tensorData, tensorElements}

tensorBufferFromList :: [Float] -> IO TensorBuffer
tensorBufferFromList values = do
    let tensorElements = length values
    tensorData <- mallocForeignPtrArray tensorElements
    withForeignPtr tensorData $ \tensorPtr ->
        pokeArray tensorPtr (realToFrac <$> values)
    pure TensorBuffer{tensorData, tensorElements}

readTensorBuffer :: TensorBuffer -> IO [Float]
readTensorBuffer tensorBuffer =
    withForeignPtr tensorBuffer.tensorData $ \tensorPtr ->
        fmap (realToFrac <$>) (peekArray tensorBuffer.tensorElements tensorPtr)

withTensorBufferPtr :: TensorBuffer -> (Ptr CFloat -> IO a) -> IO a
withTensorBufferPtr tensorBuffer = withForeignPtr tensorBuffer.tensorData
