module Runtime.CPU (
    CompiledSharedObject (..),
    compileCProgram,
    appendDispatchWrapper,
    withLoadedCPUKernel,
    runCPUKernel,
    cleanupCompiledSharedObject,
) where

import           Runtime.CPU.ABI     (appendDispatchWrapper)
import           Runtime.CPU.Execute (runCPUKernel, withLoadedCPUKernel)
import           Runtime.CPU.JIT     (CompiledSharedObject (..),
                                      cleanupCompiledSharedObject,
                                      compileCProgram)
