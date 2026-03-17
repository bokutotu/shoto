{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.CPUSpec (spec) where

import           Control.Exception (finally)
import           Runtime.CPU       (CompiledSharedObject, KernelSignature (..),
                                    appendDispatchWrapper,
                                    cleanupCompiledSharedObject,
                                    compileCProgram, parseKernelSignature,
                                    runCPUKernel, withLoadedCPUKernel)
import           Runtime.Types     (KernelArg (..), RuntimeError (..),
                                    emptyTensorBuffer, readTensorBuffer,
                                    tensorBufferFromList)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Runtime.CPU.parseKernelSignature" $ do
        it "parses a Shoto-like kernel signature" $ do
            parseKernelSignature copyKernel
                `shouldBe` Right KernelSignature{extentParamName = "N", tensorParamNames = ["A", "B"]}

        it "rejects unsupported signatures" $ do
            parseKernelSignature "int shoto_kernel(int N, float* A) { return 0; }"
                `shouldBe` Left
                    ( ErrRuntimeUnsupportedSignature
                        "expected `void shoto_kernel(int <extent>, float* <tensor>...)`"
                    )

    describe "Runtime.CPU.appendDispatchWrapper" $ do
        it "adds the dispatch wrapper for the parsed kernel" $ do
            let kernelSignature =
                    KernelSignature
                        { extentParamName = "N"
                        , tensorParamNames = ["A", "B"]
                        }
                expected =
                    copyKernel
                        <> "\n"
                        <> unlines
                            [ "void shoto_dispatch(int argc, void** args) {"
                            , "    (void)argc;"
                            , "    int* N_arg = (int*)args[0];"
                            , "    float* A_arg = (float*)args[1];"
                            , "    float* B_arg = (float*)args[2];"
                            , "    shoto_kernel(*N_arg, A_arg, B_arg);"
                            , "}"
                            ]

            appendDispatchWrapper copyKernel kernelSignature `shouldBe` expected

    describe "Runtime.CPU" $ do
        it "compiles and runs a copy kernel" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]

            actual <- withCompiledSharedObject copyKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Right ()
            actualOutput <- readTensorBuffer output
            actualOutput `shouldBe` [1, 2, 3, 4]

        it "compiles and runs a pointwise add kernel" $ do
            output <- emptyTensorBuffer 4
            lhs <- tensorBufferFromList [1, 2, 3, 4]
            rhs <- tensorBufferFromList [5, 6, 7, 8]

            actual <- withCompiledSharedObject addKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor lhs
                    , KernelArgTensor rhs
                    ]

            actual `shouldBe` Right ()
            actualOutput <- readTensorBuffer output
            actualOutput `shouldBe` [6, 8, 10, 12]

        it "reports GCC failures for malformed C" $ do
            compileCProgram "void shoto_kernel(int N, float* A) { this is not valid C; }"
                >>= (`shouldSatisfy` isGccFailure)

        it "rejects the wrong argument count before executing" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]

            actual <- withCompiledSharedObject copyKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Left (ErrRuntimeArgCountMismatch 3 4)

        it "rejects a non-integer extent argument" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]

            actual <- withCompiledSharedObject copyKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgTensor output
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Left ErrRuntimeExpectedExtentArg

        it "rejects tensor buffers that are smaller than the extent" $ do
            output <- emptyTensorBuffer 2
            input <- tensorBufferFromList [1, 2]

            actual <- withCompiledSharedObject copyKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Left (ErrRuntimeTensorTooSmall 1 4 2)

withCompiledSharedObject ::
    String ->
    (CompiledSharedObject -> IO a) ->
    IO a
withCompiledSharedObject source action = do
    compileCProgram source >>= \case
        Left err -> do
            expectationFailure $ "expected JIT compilation to succeed, but got: " <> show err
            fail "compileCProgram failed"
        Right compiledSharedObject ->
            action compiledSharedObject
                `finally` cleanupCompiledSharedObject compiledSharedObject

runCompiledKernel :: CompiledSharedObject -> [KernelArg] -> IO (Either RuntimeError ())
runCompiledKernel compiledSharedObject kernelArgs = do
    actual <- withLoadedCPUKernel compiledSharedObject $ \loadedKernel ->
        runCPUKernel loadedKernel kernelArgs
    pure $ either Left id actual

isGccFailure :: Either RuntimeError CompiledSharedObject -> Bool
isGccFailure = \case
    Left ErrRuntimeGccFailed{} -> True
    _ -> False

copyKernel :: String
copyKernel =
    unlines
        [ "void shoto_kernel(int N, float* A, float* B) {"
        , "    for (int i = 0; i < N; ++i) {"
        , "        A[i] = B[i];"
        , "    }"
        , "}"
        ]

addKernel :: String
addKernel =
    unlines
        [ "void shoto_kernel(int N, float* A, float* B, float* C) {"
        , "    for (int i = 0; i < N; ++i) {"
        , "        A[i] = B[i] + C[i];"
        , "    }"
        , "}"
        ]
