{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.CUDA.Emit (
    emitCudaProgram,
) where

import           Codegen.CUDA.Ast
import           Codegen.GenIR    (AstIterVar (..), ExtentParamName (..))
import           Data.List        (intercalate)

emitCudaProgram :: CudaProgram -> String
emitCudaProgram cudaProgram =
    unlines
        ( [ headerLine
          , indentLine 1 indexLine
          ]
            <> concatMap (renderCudaStmt 1) cudaProgram.cudaBody
            <> ["}"]
        )
  where
    headerLine =
        let CudaKernelName kernelName = cudaProgram.cudaKernelName
         in "__global__ void "
                <> kernelName
                <> "("
                <> renderParams cudaProgram.cudaExtentParam cudaProgram.cudaTensorArgs
                <> ") {"
    indexLine =
        let AstIterVar cudaIndexVar = cudaProgram.cudaIndexBinding.cudaIndexVar
         in "int "
                <> cudaIndexVar
                <> " = "
                <> renderCudaLinearIndex cudaProgram.cudaIndexBinding.cudaIndexDim
                <> ";"

renderParams :: ExtentParamName -> [CudaTensorName] -> String
renderParams (ExtentParamName extentParam) tensorArgs =
    intercalate ", " $
        ("int " <> extentParam)
            : fmap renderTensorArg tensorArgs
  where
    renderTensorArg (CudaTensorName tensorName) = "float* " <> tensorName

renderCudaLinearIndex :: CudaDim -> String
renderCudaLinearIndex dim =
    let axis =
            case dim of
                CudaX -> "x"
                CudaY -> "y"
                CudaZ -> "z"
     in "blockIdx." <> axis <> " * blockDim." <> axis <> " + threadIdx." <> axis

renderCudaStmt :: Int -> CudaStmt -> [String]
renderCudaStmt indent stmt =
    case stmt of
        CudaIfLessThan{cudaIfVar = varName, cudaIfBound = upperBound, cudaIfBody = body} ->
            let AstIterVar cudaIfVar = varName
                ExtentParamName cudaIfBound = upperBound
             in [ indentLine indent $
                    "if ("
                        <> cudaIfVar
                        <> " < "
                        <> cudaIfBound
                        <> ") {"
                ]
                    <> concatMap (renderCudaStmt (indent + 1)) body
                    <> [indentLine indent "}"]
        CudaAssign{cudaAssignTarget = target, cudaAssignExpr = expr} ->
            [indentLine indent $ renderTensorRef target <> " = " <> renderCudaExpr expr <> ";"]

renderTensorRef :: CudaTensorRef -> String
renderTensorRef ref =
    let CudaTensorName tensorName = ref.tensorName
     in tensorName
            <> concatMap (\idx -> "[" <> renderCudaExpr idx <> "]") ref.tensorIndices

renderCudaExpr :: CudaExpr -> String
renderCudaExpr expr =
    case expr of
        CudaVar (AstIterVar varName) -> varName
        CudaInt value -> show value
        CudaLoad tensorName indices ->
            let CudaTensorName tensorNameText = tensorName
             in tensorNameText
                    <> concatMap (\idx -> "[" <> renderCudaExpr idx <> "]") indices
        CudaAdd lhs rhs -> "(" <> renderCudaExpr lhs <> " + " <> renderCudaExpr rhs <> ")"
        CudaMul lhs rhs -> "(" <> renderCudaExpr lhs <> " * " <> renderCudaExpr rhs <> ")"

indentLine :: Int -> String -> String
indentLine level content = replicate (level * 4) ' ' <> content
