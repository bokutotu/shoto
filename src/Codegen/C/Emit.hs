{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.C.Emit (
    emitCProgram,
) where

import           Codegen.C.Ast
import           Codegen.GenIR (AstIterVar (..), ExtentParamName (..))
import           Data.List     (intercalate)

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
        let CFunctionName functionName = cProgram.cFunctionName
         in "void "
                <> functionName
                <> "("
                <> renderParams cProgram.cExtentParams cProgram.cTensorArgs
                <> ") {"

renderParams :: [ExtentParamName] -> [CTensorName] -> String
renderParams extentParams tensorArgs =
    intercalate ", " $
        fmap renderExtentParam extentParams <> fmap renderTensorArg tensorArgs
  where
    renderExtentParam (ExtentParamName extentParam) = "int " <> extentParam
    renderTensorArg (CTensorName tensorName) = "float* " <> tensorName

renderCStmt :: Int -> CStmt -> [String]
renderCStmt indent stmt =
    case stmt of
        CForLoop{cForVar = loopVar, cForUpperBound = upperBound, cForBody = body} ->
            let AstIterVar loopVarText = loopVar
                ExtentParamName upperBoundText = upperBound
             in [ indentLine indent $
                    "for (int "
                        <> loopVarText
                        <> " = 0; "
                        <> loopVarText
                        <> " < "
                        <> upperBoundText
                        <> "; "
                        <> loopVarText
                        <> " += 1) {"
                ]
                    <> concatMap (renderCStmt (indent + 1)) body
                    <> [indentLine indent "}"]
        CAssign{cAssignTarget = target, cAssignExpr = expr} ->
            [indentLine indent $ renderTensorRef target <> " = " <> renderCExpr expr <> ";"]

renderTensorRef :: CTensorRef -> String
renderTensorRef ref =
    let CTensorName tensorName = ref.tensorName
     in tensorName
            <> concatMap (\expr -> "[" <> renderCExpr expr <> "]") ref.tensorIndices

renderCExpr :: CExpr -> String
renderCExpr expr =
    case expr of
        CVar (AstIterVar varName) -> varName
        CInt value -> show value
        CLoad tensorName indices ->
            let CTensorName tensorNameText = tensorName
             in tensorNameText
                    <> concatMap (\idx -> "[" <> renderCExpr idx <> "]") indices
        CAdd lhs rhs -> "(" <> renderCExpr lhs <> " + " <> renderCExpr rhs <> ")"
        CMul lhs rhs -> "(" <> renderCExpr lhs <> " * " <> renderCExpr rhs <> ")"

indentLine :: Int -> String -> String
indentLine level content = replicate (level * 4) ' ' <> content
