module Codegen (
    CodegenError (..),
    generateC,
    generateCuda,
) where

import           Codegen.C.Ast       (CAstError, lowerToCProgram)
import           Codegen.C.Emit      (emitCProgram)
import           Codegen.CUDA.Ast    (CudaAstError, lowerToCudaProgram)
import           Codegen.CUDA.Emit   (emitCudaProgram)
import           Codegen.GenIR       (GenIRError, buildGenProgram)
import           Data.Bifunctor      (first)
import           FrontendIR          (Program)
import           Polyhedral.Internal (AstTree)

data CodegenError
    = CodegenGenIRError GenIRError
    | CodegenCAstError CAstError
    | CodegenCudaAstError CudaAstError
    deriving (Eq, Show)

generateC :: AstTree -> Program -> Either CodegenError String
generateC ast program = do
    genProgram <- first CodegenGenIRError $ buildGenProgram ast program
    cProgram <- first CodegenCAstError $ lowerToCProgram genProgram
    pure $ emitCProgram cProgram

generateCuda :: AstTree -> Program -> Either CodegenError String
generateCuda ast program = do
    genProgram <- first CodegenGenIRError $ buildGenProgram ast program
    cudaProgram <- first CodegenCudaAstError $ lowerToCudaProgram genProgram
    pure $ emitCudaProgram cudaProgram
