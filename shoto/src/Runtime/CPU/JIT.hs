{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.CPU.JIT (
    CompiledSharedObject (..),
    compileCProgram,
    cleanupCompiledSharedObject,
) where

import           Runtime.CPU.ABI  (appendDispatchWrapper)
import           Runtime.Types    (KernelSignature, RuntimeError (..))
import           System.Directory (doesFileExist, getTemporaryDirectory,
                                   removeFile)
import           System.Exit      (ExitCode (..))
import           System.FilePath  (replaceExtension)
import           System.IO        (hClose, openTempFile)
import           System.Process   (proc, readCreateProcessWithExitCode)

data CompiledSharedObject = CompiledSharedObject
    { cSourcePath :: FilePath
    , sharedObjectPath :: FilePath
    , kernelSignature :: KernelSignature
    }
    deriving (Eq, Show)

compileCProgram :: KernelSignature -> String -> IO (Either RuntimeError CompiledSharedObject)
compileCProgram kernelSignature source = do
    tempDir <- getTemporaryDirectory
    (cSourcePath, sourceHandle) <- openTempFile tempDir "shoto-runtime.c"
    hClose sourceHandle
    let sharedObjectPath = replaceExtension cSourcePath "so"
        wrappedSource = appendDispatchWrapper source kernelSignature
        gccArgs = optimizedSharedObjectArgs cSourcePath sharedObjectPath
    writeFile cSourcePath wrappedSource
    (exitCode, stdoutText, stderrText) <-
        readCreateProcessWithExitCode
            (proc "gcc" gccArgs)
            ""
    case exitCode of
        ExitSuccess ->
            pure $
                Right
                    CompiledSharedObject
                        { cSourcePath
                        , sharedObjectPath
                        , kernelSignature
                        }
        ExitFailure _ -> do
            cleanupPathIfExists cSourcePath
            cleanupPathIfExists sharedObjectPath
            pure $ Left $ ErrRuntimeGccFailed cSourcePath exitCode stdoutText stderrText

cleanupCompiledSharedObject :: CompiledSharedObject -> IO ()
cleanupCompiledSharedObject compiledSharedObject = do
    cleanupPathIfExists compiledSharedObject.cSourcePath
    cleanupPathIfExists compiledSharedObject.sharedObjectPath

cleanupPathIfExists :: FilePath -> IO ()
cleanupPathIfExists path = do
    pathExists <- doesFileExist path
    if pathExists
        then removeFile path
        else pure ()

optimizedSharedObjectArgs :: FilePath -> FilePath -> [String]
optimizedSharedObjectArgs cSourcePath sharedObjectPath =
    [ "-shared"
    , "-fPIC"
    , "-O3"
    , "-march=native"
    , "-mtune=native"
    , "-std=c11"
    , cSourcePath
    , "-o"
    , sharedObjectPath
    ]
