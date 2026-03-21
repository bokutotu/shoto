module Shoto (
    CompileError (..),
    DeviceConfig (..),
    CudaDim (..),
    compile,
    compileM,
) where

import           Codegen                (CodegenError, CudaDim (..), generateC,
                                         generateCuda)
import           Control.Monad.Except   (MonadError (throwError), runExceptT)
import           Control.Monad.IO.Class (MonadIO (liftIO))
import           FrontendIR             (FrontendError, Program, lowerProgram)
import           Polyhedral             (ScheduleOptimization, synthesize)
import           Polyhedral.Error       (PolyhedralError)
import           Polyhedral.Internal    (AstTree, runISL)

data DeviceConfig
    = CPU
    | GPU CudaDim
    deriving (Eq, Show)

data CompileError
    = CompileFrontendError FrontendError
    | CompilePolyhedralError PolyhedralError
    | CompileCodegenError CodegenError
    deriving (Eq, Show)

runCodegen :: DeviceConfig -> AstTree -> Program -> Either CodegenError String
runCodegen CPU ast program = generateC ast program
runCodegen (GPU dim) ast program = generateCuda dim ast program

compileM ::
    (MonadIO m, MonadError CompileError m) =>
    [ScheduleOptimization] ->
    Program ->
    DeviceConfig ->
    m String
compileM optimizations program deviceConfig = do
    raw <- either (throwError . CompileFrontendError) pure (lowerProgram program)
    islResult <- liftIO $ runISL (synthesize optimizations raw)
    ast <- either (throwError . CompilePolyhedralError) pure islResult
    either (throwError . CompileCodegenError) pure (runCodegen deviceConfig ast program)

compile :: [ScheduleOptimization] -> Program -> DeviceConfig -> IO (Either CompileError String)
compile optimizations program deviceConfig = runExceptT $ compileM optimizations program deviceConfig
