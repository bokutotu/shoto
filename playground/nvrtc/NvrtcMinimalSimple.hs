{-# LANGUAGE ForeignFunctionInterface #-}

-- Simplified version without ByteString complications
module Main where

import Foreign
import Foreign.C.Types
import Foreign.C.String
import Control.Monad (when)

-- NVRTC Types
type NvrtcProgram = Ptr ()
type NvrtcResult = CInt

-- NVRTC Result codes
nvrtcSuccess :: NvrtcResult
nvrtcSuccess = 0

-- NVRTC FFI declarations
foreign import ccall "nvrtcGetErrorString" nvrtcGetErrorString :: NvrtcResult -> IO CString
foreign import ccall "nvrtcCreateProgram" nvrtcCreateProgram :: 
    Ptr NvrtcProgram -> CString -> CString -> CInt -> Ptr CString -> Ptr CString -> IO NvrtcResult
foreign import ccall "nvrtcCompileProgram" nvrtcCompileProgram ::
    NvrtcProgram -> CInt -> Ptr CString -> IO NvrtcResult
foreign import ccall "nvrtcGetProgramLogSize" nvrtcGetProgramLogSize ::
    NvrtcProgram -> Ptr CSize -> IO NvrtcResult
foreign import ccall "nvrtcGetProgramLog" nvrtcGetProgramLog ::
    NvrtcProgram -> CString -> IO NvrtcResult
foreign import ccall "nvrtcGetPTXSize" nvrtcGetPTXSize ::
    NvrtcProgram -> Ptr CSize -> IO NvrtcResult
foreign import ccall "nvrtcGetPTX" nvrtcGetPTX ::
    NvrtcProgram -> CString -> IO NvrtcResult
foreign import ccall "nvrtcDestroyProgram" nvrtcDestroyProgram ::
    Ptr NvrtcProgram -> IO NvrtcResult

-- Simple kernel source
simpleKernel :: String
simpleKernel = unlines
    [ "extern \"C\" __global__ void simple_kernel() {"
    , "    return;"
    , "}"
    ]

main :: IO ()
main = do
    putStrLn "=== NVRTC Minimal Sample (Haskell - Simple Version) ==="
    
    -- Create NVRTC program
    alloca $ \progPtr -> do
        withCString simpleKernel $ \kernelCStr ->
            withCString "simple.cu" $ \nameCStr -> do
                result <- nvrtcCreateProgram progPtr kernelCStr nameCStr 0 nullPtr nullPtr
                
                if result /= nvrtcSuccess
                    then do
                        errStr <- nvrtcGetErrorString result >>= peekCString
                        putStrLn $ "Failed to create program: " ++ errStr
                    else do
                        prog <- peek progPtr
                        putStrLn "✓ Program created"
                        
                        -- Try to compile without any options
                        compileResult <- nvrtcCompileProgram prog 0 nullPtr
                        
                        -- Always get compilation log
                        alloca $ \logSizePtr -> do
                            _ <- nvrtcGetProgramLogSize prog logSizePtr
                            logSize <- peek logSizePtr
                            
                            when (logSize > 1) $ do
                                allocaBytes (fromIntegral logSize) $ \logPtr -> do
                                    _ <- nvrtcGetProgramLog prog logPtr
                                    log <- peekCString logPtr
                                    putStrLn $ "Compilation log:\n" ++ log
                        
                        if compileResult == nvrtcSuccess
                            then do
                                putStrLn "✓ Compilation successful"
                                
                                -- Get PTX size
                                alloca $ \ptxSizePtr -> do
                                    _ <- nvrtcGetPTXSize prog ptxSizePtr
                                    ptxSize <- peek ptxSizePtr
                                    putStrLn $ "✓ PTX size: " ++ show ptxSize ++ " bytes"
                            else do
                                errStr <- nvrtcGetErrorString compileResult >>= peekCString
                                putStrLn $ "Compilation failed: " ++ errStr
                        
                        -- Cleanup
                        _ <- nvrtcDestroyProgram progPtr
                        putStrLn "✓ Program destroyed"