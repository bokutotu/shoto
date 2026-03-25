{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.CUDA.Emit (
    emitCudaProgram,
) where

import           Codegen.CUDA.Ast
import           Data.List        (intercalate)
import           IR.Name          (ParamName, TensorName, iterNameToString,
                                   kernelNameToString, paramNameToString,
                                   tensorNameToString)
import           IR.Tensor        (TensorRef (..))

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
        "extern \"C\" __global__ void "
            <> kernelNameToString cudaProgram.cudaKernelName
            <> "("
            <> renderParams cudaProgram.cudaExtentParams cudaProgram.cudaTensorArgs
            <> ") {"

renderParams :: [ParamName] -> [TensorName] -> String
renderParams extentParams tensorArgs =
    intercalate ", " $
        fmap renderExtentParam extentParams
            <> fmap renderTensorArg tensorArgs
  where
    renderExtentParam extentParam = "int " <> paramNameToString extentParam
    renderTensorArg tensorName = "float* " <> tensorNameToString tensorName

renderCudaIndexBinding :: CudaIndexBinding -> [String]
renderCudaIndexBinding binding =
    [ indentLine 1 $
        "int "
            <> iterNameToString binding.cudaIndexVar
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
        iterNameToString varName <> " < " <> paramNameToString boundName

    renderGroupedCondition binding =
        "(" <> renderAtomicCondition binding <> ")"

renderTensorRef :: TensorRef CudaExpr -> String
renderTensorRef ref =
    tensorNameToString ref.tensorName
        <> concatMap (\idx -> "[" <> renderCudaExpr idx <> "]") ref.tensorIndices

renderCudaExpr :: CudaExpr -> String
renderCudaExpr expr =
    case expr of
        CudaVar varName -> iterNameToString varName
        CudaExtentVar extentParam -> paramNameToString extentParam
        CudaInt value -> show value
        CudaLoad tensorName indices ->
            tensorNameToString tensorName
                <> concatMap (\idx -> "[" <> renderCudaExpr idx <> "]") indices
        CudaAdd lhs rhs -> "(" <> renderCudaExpr lhs <> " + " <> renderCudaExpr rhs <> ")"
        CudaMul lhs rhs -> "(" <> renderCudaExpr lhs <> " * " <> renderCudaExpr rhs <> ")"

indentLine :: Int -> String -> String
indentLine level content = replicate (level * 4) ' ' <> content
