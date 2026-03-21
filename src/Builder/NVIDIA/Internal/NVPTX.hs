module Builder.NVIDIA.Internal.NVPTX (
    compileProgramToPtx,
) where

import           Builder.NVIDIA.Internal.NVRTC.FFI
import           Builder.Types                     (BuilderError (..))
import           Control.Exception                 (SomeException, bracket, try)
import qualified Data.ByteString                   as BS
import           Data.List                         (isPrefixOf, nub)
import           Foreign.C.String                  (CString, peekCString,
                                                    withCString)
import           Foreign.C.Types                   (CInt (..))
import           Foreign.Marshal.Alloc             (alloca, allocaBytes)
import           Foreign.Marshal.Array             (withArrayLen)
import           Foreign.Marshal.Utils             (with)
import           Foreign.Ptr                       (Ptr, nullPtr)
import           Foreign.Storable                  (peek)
import           System.Directory                  (doesDirectoryExist,
                                                    listDirectory)
import           System.Environment                (lookupEnv, setEnv)
import           System.FilePath                   ((</>))
import           System.Posix.DynamicLinker        (DL,
                                                    RTLDFlags (RTLD_GLOBAL, RTLD_NOW),
                                                    dlopen)

compileProgramToPtx :: String -> String -> [String] -> IO (Either BuilderError BS.ByteString)
compileProgramToPtx = compileProgramToPtxIO

nvrtcErrorFromResult :: String -> NvrtcResult -> Maybe String -> IO BuilderError
nvrtcErrorFromResult fnName result compileLog = do
    messagePtr <- c_nvrtcGetErrorString result
    cudaMessage <-
        if messagePtr == nullPtr
            then pure Nothing
            else Just <$> peekCString messagePtr
    pure $
        ErrBuilderCudaNvrtcError
            fnName
            (fromIntegral result)
            cudaMessage
            compileLog

compileProgramToPtxIO :: String -> String -> [String] -> IO (Either BuilderError BS.ByteString)
compileProgramToPtxIO programName source compileOptions =
    do
        ensureNvrtcBuiltinsLoaded
        withCreatedProgram programName source $ \rawProgram -> do
            compileResult <- withCStringArray compileOptions $ uncurry (c_nvrtcCompileProgram rawProgram)
            if compileResult /= nvrtcSuccess
                then do
                    compileLog <- readProgramLog rawProgram
                    Left <$> nvrtcErrorFromResult "nvrtcCompileProgram" compileResult compileLog
                else readPtx rawProgram

withCreatedProgram ::
    String ->
    String ->
    (RawProgram -> IO (Either BuilderError a)) ->
    IO (Either BuilderError a)
withCreatedProgram programName source continue = do
    createResult <- createProgram programName source
    case createResult of
        Left err -> pure $ Left err
        Right rawProgram ->
            bracket (pure rawProgram) destroyProgram continue

createProgram :: String -> String -> IO (Either BuilderError RawProgram)
createProgram programName source =
    withCString source $ \sourcePtr ->
        withCString programName $ \namePtr ->
            alloca $ \programPtr -> do
                result <- c_nvrtcCreateProgram programPtr sourcePtr namePtr 0 nullPtr nullPtr
                if result /= nvrtcSuccess
                    then do
                        compileError <- nvrtcErrorFromResult "nvrtcCreateProgram" result Nothing
                        pure $ Left compileError
                    else Right <$> peek programPtr

destroyProgram :: RawProgram -> IO ()
destroyProgram rawProgram =
    with rawProgram $ \programPtr -> do
        _ <- c_nvrtcDestroyProgram programPtr
        pure ()

readProgramLog :: RawProgram -> IO (Maybe String)
readProgramLog rawProgram = do
    result <- alloca $ \sizePtr -> do
        logSizeResult <- c_nvrtcGetProgramLogSize rawProgram sizePtr
        logSize <- peek sizePtr
        pure (logSizeResult, logSize)
    case result of
        (logSizeResult, _)
            | logSizeResult /= nvrtcSuccess -> pure Nothing
        (_, logSize)
            | logSize <= 1 -> pure Nothing
        (_, logSize) -> do
            logBytes <- allocaBytes (fromIntegral logSize) $ \logPtr -> do
                _ <- c_nvrtcGetProgramLog rawProgram logPtr
                BS.packCStringLen (logPtr, fromIntegral logSize - 1)
            pure $ Just $ bytesToString logBytes

readPtx :: RawProgram -> IO (Either BuilderError BS.ByteString)
readPtx rawProgram = do
    (sizeResult, ptxSize) <-
        alloca $ \sizePtr -> do
            sizeResult <- c_nvrtcGetPTXSize rawProgram sizePtr
            ptxSize <- peek sizePtr
            pure (sizeResult, ptxSize)
    if sizeResult /= nvrtcSuccess
        then do
            compileError <- nvrtcErrorFromResult "nvrtcGetPTXSize" sizeResult Nothing
            pure $ Left compileError
        else allocaBytes (fromIntegral ptxSize) $ \ptxPtr -> do
            ptxResult <- c_nvrtcGetPTX rawProgram ptxPtr
            if ptxResult /= nvrtcSuccess
                then do
                    compileError <- nvrtcErrorFromResult "nvrtcGetPTX" ptxResult Nothing
                    pure $ Left compileError
                else Right <$> BS.packCStringLen (ptxPtr, fromIntegral ptxSize - 1)

withCStringArray :: [String] -> ((CInt, Ptr CString) -> IO a) -> IO a
withCStringArray values continue =
    go values []
  where
    go [] reversedValues =
        withArrayLen (reverse reversedValues) $ \valueCount valueArray ->
            continue (fromIntegral valueCount, valueArray)
    go (value : remainingValues) reversedValues =
        withCString value $ \valuePtr ->
            go remainingValues (valuePtr : reversedValues)

bytesToString :: BS.ByteString -> String
bytesToString = map (toEnum . fromEnum) . BS.unpack

ensureNvrtcBuiltinsLoaded :: IO ()
ensureNvrtcBuiltinsLoaded = do
    builtinNames <- nvrtcBuiltinNames
    builtinSearchRoots <- nvrtcBuiltinSearchRoots
    prependSearchPath "LD_LIBRARY_PATH" builtinSearchRoots
    prependSearchPath "LIBRARY_PATH" builtinSearchRoots
    let builtinCandidates =
            builtinNames
                <> [builtinRoot </> builtinName | builtinRoot <- builtinSearchRoots, builtinName <- builtinNames]
    _ <- tryLoadBuiltin builtinCandidates
    pure ()

nvrtcBuiltinNames :: IO [FilePath]
nvrtcBuiltinNames =
    alloca $ \majorPtr ->
        alloca $ \minorPtr -> do
            versionResult <- c_nvrtcVersion majorPtr minorPtr
            if versionResult /= nvrtcSuccess
                then pure ["libnvrtc-builtins.so"]
                else do
                    major <- peek majorPtr
                    minor <- peek minorPtr
                    pure
                        [ "libnvrtc-builtins.so." <> show major <> "." <> show minor
                        , "libnvrtc-builtins.so"
                        ]

nvrtcBuiltinSearchRoots :: IO [FilePath]
nvrtcBuiltinSearchRoots = do
    cudaPath <- lookupEnv "CUDA_PATH"
    ldLibraryPath <- lookupEnv "LD_LIBRARY_PATH"
    versionedCudaRoots <- findVersionedCudaRoots
    pure $
        nub $
            concat
                [ maybe [] cudaCandidateRoots cudaPath
                , splitSearchPath ldLibraryPath
                , concatMap cudaCandidateRoots ("/usr/local/cuda" : versionedCudaRoots)
                ]

cudaCandidateRoots :: FilePath -> [FilePath]
cudaCandidateRoots cudaRoot =
    [ cudaRoot
    , cudaRoot </> "lib"
    , cudaRoot </> "lib64"
    , cudaRoot </> "targets" </> "x86_64-linux" </> "lib"
    ]

findVersionedCudaRoots :: IO [FilePath]
findVersionedCudaRoots = do
    let usrLocalRoot = "/usr/local"
    usrLocalExists <- doesDirectoryExist usrLocalRoot
    if not usrLocalExists
        then pure []
        else do
            usrLocalEntries <- listDirectory usrLocalRoot
            pure
                [ usrLocalRoot </> usrLocalEntry
                | usrLocalEntry <- usrLocalEntries
                , "cuda-" `isPrefixOf` usrLocalEntry
                ]

splitSearchPath :: Maybe String -> [FilePath]
splitSearchPath maybePathValue =
    case maybePathValue of
        Nothing -> []
        Just pathValue -> filter (not . null) (splitOn ':' pathValue)

splitOn :: Char -> String -> [String]
splitOn delimiter input =
    case break (== delimiter) input of
        (chunk, []) -> [chunk]
        (chunk, _ : remainingInput) -> chunk : splitOn delimiter remainingInput

tryLoadBuiltin :: [FilePath] -> IO (Maybe DL)
tryLoadBuiltin [] = pure Nothing
tryLoadBuiltin (builtinCandidate : remainingCandidates) = do
    loadResult <- try (dlopen builtinCandidate [RTLD_NOW, RTLD_GLOBAL]) :: IO (Either SomeException DL)
    case loadResult of
        Right handle -> pure $ Just handle
        Left _ -> tryLoadBuiltin remainingCandidates

prependSearchPath :: String -> [FilePath] -> IO ()
prependSearchPath envVarName searchRoots = do
    let uniqueSearchRoots = nub (filter (not . null) searchRoots)
    currentValue <- lookupEnv envVarName
    let currentRoots = splitSearchPath currentValue
        mergedRoots = nub (uniqueSearchRoots <> currentRoots)
    setEnv envVarName (joinSearchPath mergedRoots)

joinSearchPath :: [FilePath] -> String
joinSearchPath [] = ""
joinSearchPath [singlePath] = singlePath
joinSearchPath (pathHead : remainingPaths) = pathHead <> ":" <> joinSearchPath remainingPaths
