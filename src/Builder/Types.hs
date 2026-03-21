module Builder.Types (
    BuilderError (..),
) where

import           System.Exit (ExitCode)

data BuilderError
    = ErrBuilderGccFailed FilePath ExitCode String String
    | ErrBuilderCudaDriverError String Int (Maybe String) (Maybe String)
    | ErrBuilderCudaNvrtcError String Int (Maybe String) (Maybe String)
    deriving (Eq, Show)
