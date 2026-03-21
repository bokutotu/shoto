{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.CPU.ABI (
    appendDispatchWrapper,
) where

import           Data.List     (dropWhileEnd, intercalate)
import           Runtime.Types (KernelSignature (..))

appendDispatchWrapper :: String -> KernelSignature -> String
appendDispatchWrapper source kernelSignature =
    dropWhileEnd (== '\n') source
        <> "\n\n"
        <> unlines
            ( [ "void shoto_dispatch(int argc, void** args) {"
              , "    (void)argc;"
              ]
                <> extentBinding
                <> tensorBindings
                <> [ "    shoto_kernel("
                        <> intercalate ", " (("*" <> kernelSignature.extentParamName <> "_arg") : tensorArgNames)
                        <> ");"
                   , "}"
                   ]
            )
  where
    extentBinding =
        [ "    int* "
            <> kernelSignature.extentParamName
            <> "_arg = (int*)args[0];"
        ]

    tensorBindings =
        zipWith renderTensorBinding [1 :: Int ..] kernelSignature.tensorParamNames

    tensorArgNames =
        (<> "_arg") <$> kernelSignature.tensorParamNames

    renderTensorBinding argIndex tensorName =
        "    float* "
            <> tensorName
            <> "_arg = (float*)args["
            <> show argIndex
            <> "];"
