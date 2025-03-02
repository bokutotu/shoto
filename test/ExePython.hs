module ExePython where

import System.Process

-- | Execute Python script
-- >>> exePython "print('Hello, Python!')"
-- "Hello, Python!\n"
exePython :: String -> String -> IO String
exePython env script = readProcess "pyenv exec python" ["-c", script] env
