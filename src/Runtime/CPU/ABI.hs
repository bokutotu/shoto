{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.CPU.ABI (
    appendDispatchWrapper,
) where

import           Data.List     (dropWhileEnd, intercalate)
import           Runtime.Types (KernelSignature (..), KernelTensorParam (..))

appendDispatchWrapper :: String -> KernelSignature -> String
appendDispatchWrapper source kernelSignature =
    dropWhileEnd (== '\n') source
        <> "\n\n"
        <> unlines
            ( [ "void shoto_dispatch(int argc, void** args) {"
              , "    (void)argc;"
              ]
                <> extentBindings
                <> tensorBindings
                <> [ "    shoto_kernel("
                        <> intercalate ", " (extentArgNames <> tensorArgNames)
                        <> ");"
                   , "}"
                   ]
            )
  where
    extentBindings =
        zipWith renderExtentBinding [0 :: Int ..] kernelSignature.extentParamNames

    tensorBindings =
        zipWith
            renderTensorBinding
            [length kernelSignature.extentParamNames ..]
            kernelSignature.tensorParams

    tensorArgNames =
        (<> "_arg") . tensorParamName <$> kernelSignature.tensorParams

    extentArgNames =
        (\extentParamName -> "*" <> extentParamName <> "_arg") <$> kernelSignature.extentParamNames

    renderExtentBinding argIndex extentParamName =
        "    int* "
            <> extentParamName
            <> "_arg = (int*)args["
            <> show argIndex
            <> "];"

    renderTensorBinding argIndex tensorParam =
        "    float* "
            <> tensorParam.tensorParamName
            <> "_arg = (float*)args["
            <> show argIndex
            <> "];"
