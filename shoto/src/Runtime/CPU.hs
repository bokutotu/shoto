module Runtime.CPU (
    KernelSignature (..),
    CompiledSharedObject (..),
    compileCProgram,
    parseKernelSignature,
    appendDispatchWrapper,
    withLoadedCPUKernel,
    runCPUKernel,
    cleanupCompiledSharedObject,
) where

import           Runtime.CPU.ABI     (KernelSignature (..),
                                      appendDispatchWrapper,
                                      parseKernelSignature)
import           Runtime.CPU.Execute (runCPUKernel, withLoadedCPUKernel)
import           Runtime.CPU.JIT     (CompiledSharedObject (..),
                                      cleanupCompiledSharedObject,
                                      compileCProgram)
