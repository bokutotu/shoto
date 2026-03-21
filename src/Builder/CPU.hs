{-# LANGUAGE OverloadedRecordDot #-}

module Builder.CPU (
    CompiledSharedObject (..),
    compileCProgram,
    appendDispatchWrapper,
    cleanupCompiledSharedObject,
) where

import           Builder.CPU.Types (CompiledSharedObject (..))
import           Builder.Types     (BuilderError (..))
import           Control.Monad     (when)
import           Data.Foldable     (traverse_)
import           Runtime.CPU.ABI   (appendDispatchWrapper)
import           Runtime.Types     (KernelSignature)
import           System.Directory  (doesFileExist, getTemporaryDirectory,
                                    removeFile)
import           System.Exit       (ExitCode (..))
import           System.FilePath   (replaceExtension)
import           System.IO         (hClose, openTempFile)
import           System.Process    (proc, readCreateProcessWithExitCode)

compileCProgram :: KernelSignature -> String -> IO (Either BuilderError CompiledSharedObject)
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
            pure $ Left $ ErrBuilderGccFailed cSourcePath exitCode stdoutText stderrText

cleanupCompiledSharedObject :: CompiledSharedObject -> IO ()
cleanupCompiledSharedObject compiledSharedObject = do
    traverse_
        cleanupPathIfExists
        [ compiledSharedObject.cSourcePath
        , compiledSharedObject.sharedObjectPath
        ]

cleanupPathIfExists :: FilePath -> IO ()
cleanupPathIfExists path = do
    pathExists <- doesFileExist path
    when pathExists $
        removeFile path

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
