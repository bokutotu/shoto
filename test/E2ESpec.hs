{-# LANGUAGE OverloadedStrings #-}

module E2ESpec (spec) where

import qualified Builder.CPU        as BuilderCPU
import qualified Builder.NVIDIA     as BuilderNVIDIA
import           Control.Exception  (finally)
import           Control.Monad      (join)
import qualified Data.List.NonEmpty as NE
import           FrontendIR         (Axis (..), Expr (..), IxExpr (..),
                                     Program (..), Stmt (..), TensorDecl (..))
import qualified Runtime.CPU        as RuntimeCPU
import qualified Runtime.NVIDIA     as RuntimeNVIDIA
import           Runtime.Types      (KernelArg (..), KernelSignature (..),
                                     KernelTensorParam (..), RuntimeError,
                                     ThreadBlockShape (..), emptyTensorBuffer,
                                     readTensorBuffer, tensorBufferFromList)
import           Shoto              (DeviceConfig (..), compile)
import           Test.Hspec

spec :: Spec
spec =
    describe "Shoto end-to-end" $ do
        it "compiles, builds, and executes a CPU kernel" $ do
            output <- emptyTensorBuffer 4
            lhs <- tensorBufferFromList [1, 2, 3, 4]
            rhs <- tensorBufferFromList [5, 6, 7, 8]
            source <- expectCompiledSource CPU pointwiseAddProgram

            actual <-
                withCompiledSharedObject pointwiseAddKernelSignature source $ \compiledSharedObject ->
                    runCompiledKernel
                        compiledSharedObject
                        [ KernelArgInt 4
                        , KernelArgTensor output
                        , KernelArgTensor lhs
                        , KernelArgTensor rhs
                        ]

            actual `shouldBe` Right ()
            readTensorBuffer output `shouldReturn` [6, 8, 10, 12]

        it "compiles, builds, and executes a NVIDIA kernel" $ do
            output <- emptyTensorBuffer 4
            lhs <- tensorBufferFromList [1, 2, 3, 4]
            rhs <- tensorBufferFromList [5, 6, 7, 8]
            source <- expectCompiledSource GPU pointwiseAddProgram
            compiledKernel <- expectCompiledCudaProgram pointwiseAddKernelSignature source

            actual <- RuntimeNVIDIA.runNVIDIA $ do
                loadedKernel <- RuntimeNVIDIA.loadNvidiaKernel compiledKernel
                RuntimeNVIDIA.runNvidiaKernelWithHostBuffers
                    loadedKernel
                    ThreadBlockShape{blockDimX = 64, blockDimY = 1, blockDimZ = 1}
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor lhs
                    , KernelArgTensor rhs
                    ]

            actual `shouldBe` Right ()
            readTensorBuffer output `shouldReturn` [6, 8, 10, 12]

        it "compiles, builds, and executes a 2D CPU kernel" $ do
            output <- emptyTensorBuffer 6
            lhs <- tensorBufferFromList [1, 2, 3, 4, 5, 6]
            rhs <- tensorBufferFromList [10, 20, 30, 40, 50, 60]
            source <- expectCompiledSource CPU pointwiseAdd2DProgram

            actual <-
                withCompiledSharedObject pointwiseAdd2DKernelSignature source $ \compiledSharedObject ->
                    runCompiledKernel
                        compiledSharedObject
                        [ KernelArgInt 2
                        , KernelArgInt 3
                        , KernelArgTensor output
                        , KernelArgTensor lhs
                        , KernelArgTensor rhs
                        ]

            actual `shouldBe` Right ()
            readTensorBuffer output `shouldReturn` [11, 22, 33, 44, 55, 66]

expectCompiledSource :: DeviceConfig -> Program -> IO String
expectCompiledSource deviceConfig program = do
    Right source <- compile [] program deviceConfig
    pure source

withCompiledSharedObject ::
    KernelSignature ->
    String ->
    (BuilderCPU.CompiledSharedObject -> IO a) ->
    IO a
withCompiledSharedObject kernelSignature source action = do
    Right compiledSharedObject <- BuilderCPU.compileCProgram kernelSignature source
    action compiledSharedObject
        `finally` BuilderCPU.cleanupCompiledSharedObject compiledSharedObject

expectCompiledCudaProgram ::
    KernelSignature ->
    String ->
    IO BuilderNVIDIA.CompiledCudaProgram
expectCompiledCudaProgram kernelSignature source = do
    Right compiledProgram <- BuilderNVIDIA.compileCudaProgram kernelSignature source
    pure compiledProgram

runCompiledKernel ::
    BuilderCPU.CompiledSharedObject -> [KernelArg] -> IO (Either RuntimeError ())
runCompiledKernel compiledSharedObject kernelArgs = do
    actual <- RuntimeCPU.withLoadedCPUKernel compiledSharedObject $ \loadedKernel ->
        RuntimeCPU.runCPUKernel loadedKernel kernelArgs
    pure $ join actual

pointwiseAddKernelSignature :: KernelSignature
pointwiseAddKernelSignature =
    KernelSignature
        { extentParamNames = ["N"]
        , tensorParams =
            [ KernelTensorParam{tensorParamName = "A", tensorShapeParamNames = ["N"]}
            , KernelTensorParam{tensorParamName = "B", tensorShapeParamNames = ["N"]}
            , KernelTensorParam{tensorParamName = "C", tensorShapeParamNames = ["N"]}
            ]
        }

pointwiseAddProgram :: Program
pointwiseAddProgram =
    Program
        { axes =
            NE.fromList
                [Axis{iter = "i", extent = "N"}]
        , tensors =
            NE.fromList
                [ TensorDecl{tensor = "A", shape = ["N"]}
                , TensorDecl{tensor = "B", shape = ["N"]}
                , TensorDecl{tensor = "C", shape = ["N"]}
                ]
        , stmts =
            NE.fromList
                [ Assign
                    { outputTensor = "A"
                    , outputIndex = [IxVar "i"]
                    , rhs =
                        EAdd
                            (ELoad "B" [IxVar "i"])
                            (ELoad "C" [IxVar "i"])
                    }
                ]
        }

pointwiseAdd2DKernelSignature :: KernelSignature
pointwiseAdd2DKernelSignature =
    KernelSignature
        { extentParamNames = ["N", "M"]
        , tensorParams =
            [ KernelTensorParam{tensorParamName = "A", tensorShapeParamNames = ["N", "M"]}
            , KernelTensorParam{tensorParamName = "B", tensorShapeParamNames = ["N", "M"]}
            , KernelTensorParam{tensorParamName = "C", tensorShapeParamNames = ["N", "M"]}
            ]
        }

pointwiseAdd2DProgram :: Program
pointwiseAdd2DProgram =
    Program
        { axes =
            NE.fromList
                [ Axis{iter = "i", extent = "N"}
                , Axis{iter = "j", extent = "M"}
                ]
        , tensors =
            NE.fromList
                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                , TensorDecl{tensor = "B", shape = ["N", "M"]}
                , TensorDecl{tensor = "C", shape = ["N", "M"]}
                ]
        , stmts =
            NE.fromList
                [ Assign
                    { outputTensor = "A"
                    , outputIndex = [IxVar "i", IxVar "j"]
                    , rhs =
                        EAdd
                            (ELoad "B" [IxVar "i", IxVar "j"])
                            (ELoad "C" [IxVar "i", IxVar "j"])
                    }
                ]
        }
