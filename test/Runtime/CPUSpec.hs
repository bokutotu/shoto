{-# LANGUAGE LambdaCase #-}

module Runtime.CPUSpec (spec) where

import           Builder.CPU       (CompiledSharedObject, appendDispatchWrapper,
                                    cleanupCompiledSharedObject,
                                    compileCProgram)
import           Builder.Types     (BuilderError (..))
import           Control.Exception (finally)
import           Control.Monad     (join)
import           Runtime.CPU       (runCPUKernel, withLoadedCPUKernel)
import           Runtime.Types     (KernelArg (..), KernelSignature (..),
                                    KernelTensorParam (..), RuntimeError (..),
                                    emptyTensorBuffer, readTensorBuffer,
                                    tensorBufferFromList)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Runtime.CPU.appendDispatchWrapper" $ do
        it "adds the dispatch wrapper for the explicit kernel signature" $ do
            let expected =
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

            appendDispatchWrapper copyKernel copyKernelSignature `shouldBe` expected

        it "adds the dispatch wrapper for multiple extent parameters" $ do
            let expected =
                    copy2DKernel
                        <> "\n"
                        <> unlines
                            [ "void shoto_dispatch(int argc, void** args) {"
                            , "    (void)argc;"
                            , "    int* N_arg = (int*)args[0];"
                            , "    int* M_arg = (int*)args[1];"
                            , "    float* A_arg = (float*)args[2];"
                            , "    float* B_arg = (float*)args[3];"
                            , "    shoto_kernel(*N_arg, *M_arg, A_arg, B_arg);"
                            , "}"
                            ]

            appendDispatchWrapper copy2DKernel copy2DKernelSignature `shouldBe` expected

    describe "Runtime.CPU" $ do
        it "compiles and runs a copy kernel" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]

            actual <- withCompiledSharedObject copyKernelSignature copyKernel $ \compiledSharedObject ->
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

            actual <- withCompiledSharedObject addKernelSignature addKernel $ \compiledSharedObject ->
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

        it "compiles and runs a kernel without parsing its C signature" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]

            actual <- withCompiledSharedObject copyKernelSignature restrictCopyKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Right ()
            actualOutput <- readTensorBuffer output
            actualOutput `shouldBe` [1, 2, 3, 4]

        it "reports GCC failures for malformed C" $ do
            compileCProgram
                malformedKernelSignature
                "void shoto_kernel(int N, float* A) { this is not valid C; }"
                >>= (`shouldSatisfy` isGccFailure)

        it "rejects the wrong argument count before executing" $ do
            output <- emptyTensorBuffer 4
            input <- tensorBufferFromList [1, 2, 3, 4]

            actual <- withCompiledSharedObject copyKernelSignature copyKernel $ \compiledSharedObject ->
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

            actual <- withCompiledSharedObject copyKernelSignature copyKernel $ \compiledSharedObject ->
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

            actual <- withCompiledSharedObject copyKernelSignature copyKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Left (ErrRuntimeTensorTooSmall 1 4 2)

        it "compiles and runs a 2D copy kernel with flat buffers" $ do
            output <- emptyTensorBuffer 6
            input <- tensorBufferFromList [1, 2, 3, 4, 5, 6]

            actual <- withCompiledSharedObject copy2DKernelSignature copy2DKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 2
                    , KernelArgInt 3
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Right ()
            actualOutput <- readTensorBuffer output
            actualOutput `shouldBe` [1, 2, 3, 4, 5, 6]

        it "rejects buffers smaller than the multi-extent element product" $ do
            output <- emptyTensorBuffer 5
            input <- tensorBufferFromList [1, 2, 3, 4, 5]

            actual <- withCompiledSharedObject copy2DKernelSignature copy2DKernel $ \compiledSharedObject ->
                runCompiledKernel
                    compiledSharedObject
                    [ KernelArgInt 2
                    , KernelArgInt 3
                    , KernelArgTensor output
                    , KernelArgTensor input
                    ]

            actual `shouldBe` Left (ErrRuntimeTensorTooSmall 1 6 5)

withCompiledSharedObject ::
    KernelSignature ->
    String ->
    (CompiledSharedObject -> IO a) ->
    IO a
withCompiledSharedObject kernelSignature source action = do
    Right compiledSharedObject <- compileCProgram kernelSignature source
    action compiledSharedObject
        `finally` cleanupCompiledSharedObject compiledSharedObject

runCompiledKernel :: CompiledSharedObject -> [KernelArg] -> IO (Either RuntimeError ())
runCompiledKernel compiledSharedObject kernelArgs = do
    actual <- withLoadedCPUKernel compiledSharedObject $ \loadedKernel ->
        runCPUKernel loadedKernel kernelArgs
    pure $ join actual

isGccFailure :: Either BuilderError CompiledSharedObject -> Bool
isGccFailure = \case
    Left ErrBuilderGccFailed{} -> True
    _ -> False

copyKernelSignature :: KernelSignature
copyKernelSignature =
    KernelSignature
        { extentParamNames = ["N"]
        , tensorParams =
            [ KernelTensorParam{tensorParamName = "A", tensorShapeParamNames = ["N"]}
            , KernelTensorParam{tensorParamName = "B", tensorShapeParamNames = ["N"]}
            ]
        }

copyKernel :: String
copyKernel =
    unlines
        [ "void shoto_kernel(int N, float* A, float* B) {"
        , "    for (int i = 0; i < N; ++i) {"
        , "        A[i] = B[i];"
        , "    }"
        , "}"
        ]

copy2DKernelSignature :: KernelSignature
copy2DKernelSignature =
    KernelSignature
        { extentParamNames = ["N", "M"]
        , tensorParams =
            [ KernelTensorParam{tensorParamName = "A", tensorShapeParamNames = ["N", "M"]}
            , KernelTensorParam{tensorParamName = "B", tensorShapeParamNames = ["N", "M"]}
            ]
        }

copy2DKernel :: String
copy2DKernel =
    unlines
        [ "void shoto_kernel(int N, int M, float* A, float* B) {"
        , "    for (int i = 0; i < N; ++i) {"
        , "        for (int j = 0; j < M; ++j) {"
        , "            A[(i * M) + j] = B[(i * M) + j];"
        , "        }"
        , "    }"
        , "}"
        ]

addKernelSignature :: KernelSignature
addKernelSignature =
    KernelSignature
        { extentParamNames = ["N"]
        , tensorParams =
            [ KernelTensorParam{tensorParamName = "A", tensorShapeParamNames = ["N"]}
            , KernelTensorParam{tensorParamName = "B", tensorShapeParamNames = ["N"]}
            , KernelTensorParam{tensorParamName = "C", tensorShapeParamNames = ["N"]}
            ]
        }

addKernel :: String
addKernel =
    unlines
        [ "void shoto_kernel(int N, float* A, float* B, float* C) {"
        , "    for (int i = 0; i < N; ++i) {"
        , "        A[i] = B[i] + C[i];"
        , "    }"
        , "}"
        ]

restrictCopyKernel :: String
restrictCopyKernel =
    unlines
        [ "void shoto_kernel(int N, float *restrict A, float *restrict B) {"
        , "    for (int i = 0; i < N; ++i) {"
        , "        A[i] = B[i];"
        , "    }"
        , "}"
        ]

malformedKernelSignature :: KernelSignature
malformedKernelSignature =
    KernelSignature
        { extentParamNames = ["N"]
        , tensorParams =
            [KernelTensorParam{tensorParamName = "A", tensorShapeParamNames = ["N"]}]
        }
