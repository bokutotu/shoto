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
                                     RuntimeError, emptyTensorBuffer,
                                     readTensorBuffer, tensorBufferFromList)
import           Shoto              (CudaDim (..), DeviceConfig (..), compile)
import           Test.Hspec

spec :: Spec
spec =
    describe "Shoto end-to-end" $ do
        it "compiles, builds, and executes a CPU kernel" $ do
            output <- emptyTensorBuffer 4
            lhs <- tensorBufferFromList [1, 2, 3, 4]
            rhs <- tensorBufferFromList [5, 6, 7, 8]
            source <- expectCompiledSource CPU

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
            source <- expectCompiledSource (GPU CudaX)
            compiledKernel <- expectCompiledCudaProgram pointwiseAddKernelSignature CudaX source

            actual <- RuntimeNVIDIA.runNVIDIA $ do
                loadedKernel <- RuntimeNVIDIA.loadNvidiaKernel compiledKernel
                RuntimeNVIDIA.runNvidiaKernelWithHostBuffers
                    loadedKernel
                    64
                    [ KernelArgInt 4
                    , KernelArgTensor output
                    , KernelArgTensor lhs
                    , KernelArgTensor rhs
                    ]

            actual `shouldBe` Right ()
            readTensorBuffer output `shouldReturn` [6, 8, 10, 12]

expectCompiledSource :: DeviceConfig -> IO String
expectCompiledSource deviceConfig = do
    Right source <- compile [] pointwiseAddProgram deviceConfig
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
    CudaDim ->
    String ->
    IO BuilderNVIDIA.CompiledCudaProgram
expectCompiledCudaProgram kernelSignature cudaDim source = do
    Right compiledProgram <- BuilderNVIDIA.compileCudaProgram kernelSignature cudaDim source
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
        { extentParamName = "N"
        , tensorParamNames = ["A", "B", "C"]
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
