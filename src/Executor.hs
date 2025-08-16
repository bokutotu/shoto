module Executor where

import qualified Data.ByteString.Char8 as BS
import qualified Data.Map              as M
import           Foreign               (alloca, castPtr, peek, poke,
                                        withForeignPtr)
import           Foreign.C.Types       (CInt)
import           Foreign.Ptr           (Ptr)
import           IR                    (Node (..), ValueId)
import qualified IR
import           Internal.FFI          (CUmodule, cuModuleGetFunction,
                                        cuModuleLoadData, cudaSuccess)
import           Runtime
import qualified TinyIR                as TIR

-- モジュールとファンクションのペアを保持
type CompiledKernel = (CUmodule, CUfunction)
type CompiledGraph = M.Map ValueId CompiledKernel

-- 現状は一つのOperationと2つか1つのInputがあるGraphのみを対称にする
-- TODO: 複数のOpsetへの対応
compileGraph :: TIR.TinyIR -> [ValueId] -> IO CompiledGraph
compileGraph ir outputIds = do
    initializeCuda
    case outputIds of
        [] -> pure M.empty
        (outId : _) ->
            case M.lookup outId ir of
                Just (Operation op inputs) -> do
                    -- CUDAコード生成
                    let cudaCode = generateCudaCode op inputs ir
                    
                    -- コンパイルしてモジュールとファンクションを永続的に保持
                    withCudaContext 0 $ \_ -> do
                        -- PTXにコンパイル
                        ptx <- compileCudaKernel cudaCode (BS.pack "generated.cu") []
                        
                        -- モジュールをロード（永続的に保持）
                        alloca $ \modulePtr -> do
                            BS.useAsCString ptx $ \ptxCStr -> do
                                checkCuda "cuModuleLoadData" =<< cuModuleLoadData modulePtr (castPtr ptxCStr)
                            modul <- peek modulePtr
                            
                            -- ファンクションを取得
                            alloca $ \funcPtr -> do
                                BS.useAsCString (BS.pack "kernel") $ \funcName -> do
                                    checkCuda "cuModuleGetFunction" =<< cuModuleGetFunction funcPtr modul funcName
                                func <- peek funcPtr
                                
                                return $ M.singleton outId (modul, func)
                _ -> return M.empty

-- 演算に応じたCUDAコード生成
generateCudaCode :: TIR.TinyOp -> [ValueId] -> TIR.TinyIR -> BS.ByteString
generateCudaCode (TIR.ElementWise op) inputs ir =
    -- 入力のshapeを取得（今は固定で10と仮定）
    TIR.codegenElementWise op []
generateCudaCode (TIR.Reduce op axis) inputs ir =
    TIR.codegenReduce op axis []
generateCudaCode _ _ _ = error "Unsupported operation"

executeGraph ::
    forall a. TIR.TinyIR -> CompiledGraph -> [(ValueId, GpuPtr a)] -> (ValueId, GpuPtr a) -> IO ()
executeGraph ir functions inputs (outputId, output) = do
    -- CUDAコンテキストを設定
    withCudaContext 0 $ \_ -> do
        -- コンパイル済み関数を取得
        case M.lookup outputId functions of
            Nothing -> error "Function not compiled"
            Just (modul, func) -> do  -- モジュールとファンクションのペアを受け取る
                -- 演算の入力を特定
                case M.lookup outputId ir of
                    Just (Operation _ opInputs) -> do
                        -- 入力ポインタを準備
                        let getPtr vid = lookup vid inputs
                            inputPtrs = mapM getPtr opInputs
                        case inputPtrs of
                            Just [input1, input2] -> do
                                -- 2入力の場合（Binary演算）
                                launchBinaryKernel func input1 input2 output
                            _ -> error "Unsupported input configuration"
                    _ -> error "Output not found in IR"

-- Binary演算用のカーネル起動
launchBinaryKernel :: CUfunction -> GpuPtr a -> GpuPtr a -> GpuPtr a -> IO ()
launchBinaryKernel func input1 input2 output = do
    withForeignPtr (ptr input1) $ \p1 ->
        withForeignPtr (ptr input2) $ \p2 ->
            withForeignPtr (ptr output) $ \pOut -> do
                let n = size output
                    blockSize = 256
                    gridSize = (n + blockSize - 1) `div` blockSize
                    config =
                        KernelLaunchConfig
                            { gridDimX = fromIntegral gridSize
                            , gridDimY = 1
                            , gridDimZ = 1
                            , blockDimX = fromIntegral blockSize
                            , blockDimY = 1
                            , blockDimZ = 1
                            , sharedMemBytes = 0
                            }
                -- カーネル引数の準備
                -- CUDAカーネルの引数は、値へのポインタのポインタとして渡す必要がある
                alloca $ \aPtr -> alloca $ \bPtr -> alloca $ \cPtr -> alloca $ \nPtr -> do
                    poke aPtr p1
                    poke bPtr p2
                    poke cPtr pOut
                    poke nPtr (fromIntegral n :: CInt)
                    let args = [castPtr aPtr, castPtr bPtr, castPtr cPtr, castPtr nPtr]
                    launchKernel func config args
