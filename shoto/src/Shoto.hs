module Shoto (
    CompileError (..),
    compile,
    compileM,
) where

import           Control.Monad.Except   (MonadError (throwError), runExceptT)
import           Control.Monad.IO.Class (MonadIO (liftIO))
import           FrontendIR             (FrontendError, Program, checkProgram,
                                         lowerToRaw)
import           ISL                    (AstTree, IslError, runISL)
import           Polyhedral             (ScheduleOptimization, synthesize)

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
    checked <- either (throwError . CompileFrontendError) pure (checkProgram program)
    let raw = lowerToRaw checked
    islResult <- liftIO $ runISL (synthesize optimizations raw)
    either (throwError . CompileIslError) pure islResult

compile :: [ScheduleOptimization] -> Program -> IO (Either CompileError AstTree)
compile optimizations program = runExceptT $ compileM optimizations program
