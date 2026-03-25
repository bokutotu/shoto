{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.C.Emit (
    emitCProgram,
) where

import           Codegen.C.Ast
import           Data.List     (intercalate)
import           IR.Name       (ParamName, TensorName, iterNameToString,
                                kernelNameToString, paramNameToString,
                                tensorNameToString)
import           IR.Tensor     (TensorRef (..))

emitCProgram :: CProgram -> String
emitCProgram cProgram =
    unlines
        ( [ headerLine
          ]
            <> concatMap (renderCStmt 1) cProgram.cBody
            <> ["}"]
        )
  where
    headerLine =
        "void "
            <> kernelNameToString cProgram.cFunctionName
            <> "("
            <> renderParams cProgram.cExtentParams cProgram.cTensorArgs
            <> ") {"

renderParams :: [ParamName] -> [TensorName] -> String
renderParams extentParams tensorArgs =
    intercalate ", " $
        fmap renderExtentParam extentParams <> fmap renderTensorArg tensorArgs
  where
    renderExtentParam extentParam = "int " <> paramNameToString extentParam
    renderTensorArg tensorName = "float* " <> tensorNameToString tensorName

renderCStmt :: Int -> CStmt -> [String]
renderCStmt indent stmt =
    case stmt of
        CForLoop{cForVar = loopVar, cForUpperBound = upperBound, cForBody = body} ->
            [ indentLine indent $
                "for (int "
                    <> iterNameToString loopVar
                    <> " = 0; "
                    <> iterNameToString loopVar
                    <> " < "
                    <> paramNameToString upperBound
                    <> "; "
                    <> iterNameToString loopVar
                    <> " += 1) {"
            ]
                <> concatMap (renderCStmt (indent + 1)) body
                <> [indentLine indent "}"]
        CAssign{cAssignTarget = target, cAssignExpr = expr} ->
            [indentLine indent $ renderTensorRef target <> " = " <> renderCExpr expr <> ";"]

renderTensorRef :: TensorRef CExpr -> String
renderTensorRef ref =
    tensorNameToString ref.tensorName
        <> concatMap (\expr -> "[" <> renderCExpr expr <> "]") ref.tensorIndices

renderCExpr :: CExpr -> String
renderCExpr expr =
    case expr of
        CVar varName -> iterNameToString varName
        CExtentVar paramName -> paramNameToString paramName
        CInt value -> show value
        CLoad tensorName indices ->
            tensorNameToString tensorName
                <> concatMap (\idx -> "[" <> renderCExpr idx <> "]") indices
        CAdd lhs rhs -> "(" <> renderCExpr lhs <> " + " <> renderCExpr rhs <> ")"
        CMul lhs rhs -> "(" <> renderCExpr lhs <> " * " <> renderCExpr rhs <> ")"

indentLine :: Int -> String -> String
indentLine level content = replicate (level * 4) ' ' <> content
