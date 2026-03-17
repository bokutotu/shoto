{-# LANGUAGE OverloadedRecordDot #-}

module Runtime.CPU.ABI (
    KernelSignature (..),
    parseKernelSignature,
    appendDispatchWrapper,
) where

import           Data.Char                    (isAlphaNum, isLetter, isSpace)
import           Data.List                    (dropWhileEnd, intercalate, tails)
import           Data.Maybe                   (listToMaybe, mapMaybe)
import           Runtime.Types                (RuntimeError (..))
import           Text.ParserCombinators.ReadP (ReadP, char, many, munch, munch1,
                                               readP_to_S, satisfy, skipSpaces,
                                               string)

data KernelSignature = KernelSignature
    { extentParamName :: String
    , tensorParamNames :: [String]
    }
    deriving (Eq, Show)

parseKernelSignature :: String -> Either RuntimeError KernelSignature
parseKernelSignature source =
    case firstParsedSignature of
        Just kernelSignature -> Right kernelSignature
        Nothing ->
            Left $
                ErrRuntimeUnsupportedSignature
                    "expected `void shoto_kernel(int <extent>, float* <tensor>...)`"
  where
    firstParsedSignature = listToMaybe $ mapMaybe parsedSignatureAt (tails source)

    parsedSignatureAt suffix =
        case readP_to_S signatureParser suffix of
            [] -> Nothing
            parsed -> Just $ fst $ last parsed

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
        (\tensorName -> tensorName <> "_arg") <$> kernelSignature.tensorParamNames

    renderTensorBinding argIndex tensorName =
        "    float* "
            <> tensorName
            <> "_arg = (float*)args["
            <> show argIndex
            <> "];"

signatureParser :: ReadP KernelSignature
signatureParser = do
    skipSpaces
    _ <- string "void"
    skipRequiredSpace
    _ <- string "shoto_kernel"
    skipSpaces
    _ <- char '('
    skipSpaces
    extentParamName <- intParamParser
    tensorParamNames <- many $ do
        skipSpaces
        _ <- char ','
        skipSpaces
        floatPointerParamParser
    skipSpaces
    _ <- char ')'
    pure KernelSignature{extentParamName, tensorParamNames}

intParamParser :: ReadP String
intParamParser = do
    _ <- string "int"
    skipRequiredSpace
    identifierParser

floatPointerParamParser :: ReadP String
floatPointerParamParser = do
    _ <- string "float"
    skipSpaces
    _ <- char '*'
    skipSpaces
    identifierParser

identifierParser :: ReadP String
identifierParser = do
    leading <- satisfy isIdentifierStart
    trailing <- munch isIdentifierContinue
    pure $ leading : trailing

skipRequiredSpace :: ReadP ()
skipRequiredSpace = do
    _ <- munch1 isSpace
    pure ()

isIdentifierStart :: Char -> Bool
isIdentifierStart charValue = charValue == '_' || isLetter charValue

isIdentifierContinue :: Char -> Bool
isIdentifierContinue charValue = charValue == '_' || isAlphaNum charValue
