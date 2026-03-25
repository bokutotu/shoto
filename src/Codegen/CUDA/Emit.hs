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
          ]
            <> concatMap renderCudaIndexBinding cudaProgram.cudaIndexBindings
            <> concatMap (renderCudaStmt 1) cudaProgram.cudaBody
            <> ["}"]
        )
  where
    headerLine =
        let CudaKernelName kernelName = cudaProgram.cudaKernelName
         in "extern \"C\" __global__ void "
                <> kernelName
                <> "("
                <> renderParams cudaProgram.cudaExtentParams cudaProgram.cudaTensorArgs
                <> ") {"

renderParams :: [ExtentParamName] -> [CudaTensorName] -> String
renderParams extentParams tensorArgs =
    intercalate ", " $
        fmap renderExtentParam extentParams
            <> fmap renderTensorArg tensorArgs
  where
    renderExtentParam (ExtentParamName extentParam) = "int " <> extentParam
    renderTensorArg (CudaTensorName tensorName) = "float* " <> tensorName

renderCudaIndexBinding :: CudaIndexBinding -> [String]
renderCudaIndexBinding binding =
    let AstIterVar cudaIndexVar = binding.cudaIndexVar
     in [ indentLine 1 $
            "int "
                <> cudaIndexVar
                <> " = "
                <> renderCudaIndexLinearExpr binding
                <> ";"
        ]

renderCudaStmt :: Int -> CudaStmt -> [String]
renderCudaStmt indent stmt =
    case stmt of
        CudaIfAllLessThan{cudaIfBindings = bindings, cudaIfBody = body} ->
            [ indentLine indent $
                "if ("
                    <> renderGuard bindings
                    <> ") {"
            ]
                <> concatMap (renderCudaStmt (indent + 1)) body
                <> [indentLine indent "}"]
        CudaAssign{cudaAssignTarget = target, cudaAssignExpr = expr} ->
            [indentLine indent $ renderTensorRef target <> " = " <> renderCudaExpr expr <> ";"]
  where
    renderGuard [binding] = renderAtomicCondition binding
    renderGuard guardBindings = intercalate " && " (renderGroupedCondition <$> guardBindings)

    renderAtomicCondition (varName, boundName) =
        let AstIterVar varText = varName
            ExtentParamName boundText = boundName
         in varText <> " < " <> boundText

    renderGroupedCondition binding =
        "(" <> renderAtomicCondition binding <> ")"

renderTensorRef :: CudaTensorRef -> String
renderTensorRef ref =
    let CudaTensorName tensorName = ref.tensorName
     in tensorName
            <> concatMap (\idx -> "[" <> renderCudaExpr idx <> "]") ref.tensorIndices

renderCudaExpr :: CudaExpr -> String
renderCudaExpr expr =
    case expr of
        CudaVar (AstIterVar varName) -> varName
        CudaExtentVar (ExtentParamName extentParam) -> extentParam
        CudaInt value -> show value
        CudaLoad tensorName indices ->
            let CudaTensorName tensorNameText = tensorName
             in tensorNameText
                    <> concatMap (\idx -> "[" <> renderCudaExpr idx <> "]") indices
        CudaAdd lhs rhs -> "(" <> renderCudaExpr lhs <> " + " <> renderCudaExpr rhs <> ")"
        CudaMul lhs rhs -> "(" <> renderCudaExpr lhs <> " * " <> renderCudaExpr rhs <> ")"

indentLine :: Int -> String -> String
indentLine level content = replicate (level * 4) ' ' <> content
