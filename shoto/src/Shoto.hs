module Shoto (
    CompileError (..),
    compile,
    compileM,
) where

import           Control.Monad.Except   (MonadError (throwError), runExceptT)
import           Control.Monad.IO.Class (MonadIO (liftIO))
import           FrontendIR             (FrontendError, Program, lowerProgram)
import           Polyhedral             (ScheduleOptimization, synthesize)
import           Polyhedral.Internal    (AstTree, IslError, runISL)

data CompileError
    = CompileFrontendError FrontendError
    | CompileIslError IslError
    deriving (Eq, Show)

compileM ::
    (MonadIO m, MonadError CompileError m) =>
    [ScheduleOptimization] ->
    Program ->
    m AstTree
compileM optimizations program = do
    raw <- either (throwError . CompileFrontendError) pure (lowerProgram program)
    islResult <- liftIO $ runISL (synthesize optimizations raw)
    either (throwError . CompileIslError) pure islResult

compile :: [ScheduleOptimization] -> Program -> IO (Either CompileError AstTree)
compile optimizations program = runExceptT $ compileM optimizations program
