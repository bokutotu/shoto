module Builder.NVIDIA (
    CompiledCudaProgram (..),
    compileCudaProgram,
) where

import qualified Builder.NVIDIA.Internal.Device as CUDA
import qualified Builder.NVIDIA.Internal.NVPTX  as NVPTX
import           Builder.NVIDIA.Types           (CompiledCudaProgram (..))
import           Builder.Types                  (BuilderError)
import           Codegen.CUDA.Ast               (CudaDim)
import           Runtime.Types                  (KernelSignature)

compileCudaProgram ::
    KernelSignature -> CudaDim -> String -> IO (Either BuilderError CompiledCudaProgram)
compileCudaProgram kernelSignature compiledCudaDim source = do
    computeCapabilityResult <- CUDA.computeCapability
    case computeCapabilityResult of
        Left err -> pure $ Left err
        Right (major, minor) -> do
            let compiledPtxOptions =
                    [ "--gpu-architecture=compute_" <> show major <> show minor
                    , "--std=c++11"
                    ]
            compiledPtxResult <- NVPTX.compileProgramToPtx "shoto-runtime.cu" source compiledPtxOptions
            pure $
                fmap
                    ( \compiledPtx -> CompiledCudaProgram{compiledPtx, compiledKernelSignature = kernelSignature, compiledCudaDim}
                    )
                    compiledPtxResult
