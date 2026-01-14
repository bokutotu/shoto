{-# LANGUAGE OverloadedRecordDot #-}

module ISL.Ast.Read (
    parseSetExpr,
    parseUnionSetExpr,
    parseMapExpr,
    parseUnionMapExpr,
    parseAffineExpr,
    parsePwAffineExpr,
    parseMultiAffineExpr,
    parseMultiPwAffineExpr,
    parseMultiUnionPwAffineExpr,
    parseScheduleTree,
    parseConstraint,
) where

import           Control.Applicative        (empty, (<|>))
import           Data.Functor               (($>), (<&>))
import           Data.Map.Strict            (Map)
import qualified Data.Map.Strict            as Map
import           Data.Maybe                 (fromMaybe)
import           Data.Text                  (Text)
import qualified Data.Text                  as T
import           Data.Void                  (Void)
import           ISL.Ast.Types
import           Text.Megaparsec            (Parsec, between, eof,
                                             errorBundlePretty, many, manyTill,
                                             notFollowedBy, optional, runParser,
                                             sepBy, sepBy1, some, try)
import           Text.Megaparsec.Char       (alphaNumChar, char, letterChar,
                                             space1, string)
import qualified Text.Megaparsec.Char.Lexer as L

type Parser = Parsec Void String

parseSetExpr :: String -> Either String SetExpr
parseSetExpr = parseWith pSetExpr

parseUnionSetExpr :: String -> Either String UnionSetExpr
parseUnionSetExpr = parseWith pUnionSetExpr

parseMapExpr :: String -> Either String MapExpr
parseMapExpr = parseWith pMapExpr

parseUnionMapExpr :: String -> Either String UnionMapExpr
parseUnionMapExpr = parseWith pUnionMapExpr

parseAffineExpr :: String -> Either String AffineExpr
parseAffineExpr = parseWith pAffineExprFull

parsePwAffineExpr :: String -> Either String PwAffineExpr
parsePwAffineExpr = parseWith pPwAffineExprFull

parseMultiAffineExpr :: String -> Either String MultiAffineExpr
parseMultiAffineExpr = parseWith pMultiAffineExpr

parseMultiPwAffineExpr :: String -> Either String MultiPwAffineExpr
parseMultiPwAffineExpr = parseWith pMultiPwAffineExpr

parseMultiUnionPwAffineExpr :: String -> Either String MultiUnionPwAffineExpr
parseMultiUnionPwAffineExpr = parseWith pMultiUnionPwAffineExpr

parseScheduleTree :: String -> Either String ScheduleTree
parseScheduleTree = parseWith pScheduleTree

parseConstraint :: String -> Either String Constraint
parseConstraint = parseWith pConstraintFull

parseWith :: Parser a -> String -> Either String a
parseWith p input =
    case runParser (sc *> p <* sc <* eof) "<isl-ast>" input of
        Left err     -> Left (errorBundlePretty err)
        Right result -> Right result

sc :: Parser ()
sc = L.space space1 empty empty

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbol :: String -> Parser String
symbol = L.symbol sc

keyword :: String -> Parser String
keyword word = lexeme (string word <* notFollowedBy identChar)

identChar :: Parser Char
identChar = alphaNumChar <|> char '_'

pIdent :: Parser Text
pIdent = lexeme $ do
    first <- letterChar <|> char '_'
    rest <- many identChar
    pure $ T.pack (first : rest)

pDimList :: Parser [SpaceDim]
pDimList =
    between (symbol "[") (symbol "]") (pIdent `sepBy` symbol ",")
        <&> map (SpaceDim . Just)

pLocalList :: Parser [LocalDim]
pLocalList = pIdent `sepBy1` symbol "," <&> map (\name -> LocalDim (Just name) Nothing)

pTuple :: Parser Tuple
pTuple = try namedTuple <|> unnamedTuple
  where
    namedTuple = Tuple . Just <$> pIdent <*> pDimList
    unnamedTuple = Tuple Nothing <$> pDimList

emptyTuple :: Tuple
emptyTuple = Tuple Nothing []

mkSetSpace :: [SpaceDim] -> Tuple -> Space
mkSetSpace params tuple =
    Space
        { spaceParams = params
        , spaceIn = tuple
        , spaceOut = emptyTuple
        }

mkMapSpace :: [SpaceDim] -> Tuple -> Tuple -> Space
mkMapSpace params dom ran =
    Space
        { spaceParams = params
        , spaceIn = dom
        , spaceOut = ran
        }

pSetExpr :: Parser SetExpr
pSetExpr = do
    UnionSetExpr parts <- pUnionSetExpr
    case parts of
        [single] -> pure single
        _        -> fail "expected a single set"

pUnionSetExpr :: Parser UnionSetExpr
pUnionSetExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <- between (symbol "{") (symbol "}") (pBasicSet (fromMaybe [] params) `sepBy` symbol ";")
    pure $ UnionSetExpr (groupBasicSets parts)

pBasicSet :: [SpaceDim] -> Parser BasicSet
pBasicSet params = do
    tuple <- pTuple
    let space = mkSetSpace params tuple
    constraints <- optional (symbol ":" *> pConstraintsRaw space)
    case constraints of
        Nothing -> pure $ BasicSet (LocalSpace space []) []
        Just (locals, rawCons) ->
            let ls0 = LocalSpace space locals
                (lsFinal, cons) = convertConstraints ls0 rawCons
             in pure $ BasicSet lsFinal cons

pMapExpr :: Parser MapExpr
pMapExpr = do
    UnionMapExpr parts <- pUnionMapExpr
    case parts of
        [single] -> pure single
        _        -> fail "expected a single map"

pUnionMapExpr :: Parser UnionMapExpr
pUnionMapExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <- between (symbol "{") (symbol "}") (pBasicMap (fromMaybe [] params) `sepBy` symbol ";")
    pure $ UnionMapExpr (groupBasicMaps parts)

pBasicMap :: [SpaceDim] -> Parser BasicMap
pBasicMap params = do
    dom <- pTuple
    _ <- symbol "->"
    ran <- pTuple
    let space = mkMapSpace params dom ran
    constraints <- optional (symbol ":" *> pConstraintsRaw space)
    case constraints of
        Nothing -> pure $ BasicMap (LocalSpace space []) []
        Just (locals, rawCons) ->
            let ls0 = LocalSpace space locals
                (lsFinal, cons) = convertConstraints ls0 rawCons
             in pure $ BasicMap lsFinal cons

pConstraintsRaw :: Space -> Parser ([LocalDim], [RawConstraint])
pConstraintsRaw space = do
    existsBlock <- optional (try (pExistsBlock space))
    case existsBlock of
        Nothing -> do
            let env = DimEnv space []
            constraints <- pConstraintGroup env
            pure ([], constraints)
        Just (locals, constraints) -> do
            let env = DimEnv space locals
            rest <- optional (keyword "and" *> pConstraintGroup env)
            pure (locals, constraints ++ fromMaybe [] rest)

pConstraintGroup :: DimEnv -> Parser [RawConstraint]
pConstraintGroup env = concat <$> pConstraintChain env `sepBy1` keyword "and"

pExistsBlock :: Space -> Parser ([LocalDim], [RawConstraint])
pExistsBlock space = do
    _ <- keyword "exists"
    between (symbol "(") (symbol ")") $ do
        locals <- pLocalList
        _ <- symbol ":"
        let env = DimEnv space locals
        constraints <- pConstraintGroup env
        pure (locals, constraints)

pConstraintChain :: DimEnv -> Parser [RawConstraint]
pConstraintChain env = do
    first <- pAffineExprInline env
    rest <- some ((,) <$> pRelOp <*> pAffineExprInline env)
    let affs = first : map snd rest
        rels = map fst rest
        pairs = zip3 rels affs (drop 1 affs)
    mapM (\(op, lhs, rhs) -> buildConstraint op lhs rhs) pairs

data RelOp = OpLe | OpGe | OpEq | OpLt | OpGt

pRelOp :: Parser RelOp
pRelOp =
    (try (symbol "<=") $> OpLe)
        <|> (try (symbol ">=") $> OpGe)
        <|> (symbol "<" $> OpLt)
        <|> (symbol ">" $> OpGt)
        <|> (symbol "=" $> OpEq)

buildConstraint :: RelOp -> RawLinearExpr -> RawLinearExpr -> Parser RawConstraint
buildConstraint op lhs rhs =
    case op of
        OpLe -> pure $ RawConstraint RelLe lhs rhs
        OpGe -> pure $ RawConstraint RelGe lhs rhs
        OpEq -> pure $ RawConstraint RelEq lhs rhs
        OpLt -> pure $ RawConstraint RelLe lhs (shiftRaw (-1) rhs)
        OpGt -> pure $ RawConstraint RelGe lhs (shiftRaw 1 rhs)

shiftRaw :: Integer -> RawLinearExpr -> RawLinearExpr
shiftRaw delta raw = raw{rawConstant = raw.rawConstant + delta}

pAffineExprInline :: DimEnv -> Parser RawLinearExpr
pAffineExprInline = pLinearExpr

pAffineExprFull :: Parser AffineExpr
pAffineExprFull = do
    params <- optional (try (pDimList <* symbol "->"))
    pAffineExprFullWithParams (fromMaybe [] params)

pAffineExprFullWithParams :: [SpaceDim] -> Parser AffineExpr
pAffineExprFullWithParams params = do
    parts <- between (symbol "{") (symbol "}") (pAffinePiece params `sepBy` symbol ";")
    case parts of
        []       -> fail "expected affine expression"
        [single] -> pure single
        _        -> fail "expected a single affine expression"

pAffinePiece :: [SpaceDim] -> Parser AffineExpr
pAffinePiece params = do
    tuple <- pTuple
    let space = mkSetSpace params tuple
    _ <- symbol "->"
    raw <- pLinearExpr (DimEnv space [])
    constraints <- optional (symbol ":" *> pConstraintsRaw space)
    case constraints of
        Nothing ->
            let ls0 = LocalSpace space []
                (_, aff) = convertAffine ls0 raw
             in pure aff
        Just _ -> fail "affine expression cannot have domain constraints"

pPwAffineExprFull :: Parser PwAffineExpr
pPwAffineExprFull = do
    params <- optional (try (pDimList <* symbol "->"))
    pPwAffineExprFullWithParams (fromMaybe [] params)

pPwAffineExprFullWithParams :: [SpaceDim] -> Parser PwAffineExpr
pPwAffineExprFullWithParams params = do
    parts <- between (symbol "{") (symbol "}") (pPwAffinePiece params `sepBy` symbol ";")
    case parts of
        [] -> fail "expected piecewise affine expression"
        _ -> do
            space <- commonPieceSpace parts
            pure $ PwAffineExpr space parts

pPwAffinePiece :: [SpaceDim] -> Parser (BasicSet, AffineExpr)
pPwAffinePiece params = do
    tuple <- pTuple
    let space = mkSetSpace params tuple
    _ <- symbol "->"
    rawExpr <- pLinearExpr (DimEnv space [])
    let ls0 = LocalSpace space []
        (_, affExpr) = convertAffine ls0 rawExpr
    constraints <- optional (symbol ":" *> pConstraintsRaw space)
    basicSet <-
        case constraints of
            Nothing -> pure $ BasicSet (LocalSpace space []) []
            Just (locals, rawCons) ->
                let lsBase = LocalSpace space locals
                    (lsFinal, cons) = convertConstraints lsBase rawCons
                 in pure $ BasicSet lsFinal cons
    pure (basicSet, affExpr)

commonPieceSpace :: [(BasicSet, AffineExpr)] -> Parser Space
commonPieceSpace [] = fail "expected affine pieces"
commonPieceSpace ((basicSet, _) : rest) =
    let baseSpace = basicSet.basicSetSpace.localSpaceBase
        allSame = all ((== baseSpace) . (.localSpaceBase) . (.basicSetSpace) . fst) rest
     in if allSame
            then pure baseSpace
            else fail "piecewise affine parts have inconsistent spaces"

pConstraintFull :: Parser Constraint
pConstraintFull = do
    lhs <- pAffineExprFull
    op <- pRelOp
    rhs <- pAffineExprFull
    buildConstraintFull op lhs rhs

buildConstraintFull :: RelOp -> AffineExpr -> AffineExpr -> Parser Constraint
buildConstraintFull op lhs rhs =
    case op of
        OpLe -> pure $ Constraint RelLe lhs rhs
        OpGe -> pure $ Constraint RelGe lhs rhs
        OpEq -> pure $ Constraint RelEq lhs rhs
        OpLt -> do
            shifted <- shiftAffine (-1) rhs
            pure $ Constraint RelLe lhs shifted
        OpGt -> do
            shifted <- shiftAffine 1 rhs
            pure $ Constraint RelGe lhs shifted

shiftAffine :: Integer -> AffineExpr -> Parser AffineExpr
shiftAffine delta aff =
    let form = aff.affForm
        updated = form{linearConstant = form.linearConstant + delta}
     in pure aff{affForm = updated}

pMultiAffineExpr :: Parser MultiAffineExpr
pMultiAffineExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <- between (symbol "[") (symbol "]") (pMultiAffPart (fromMaybe [] params) `sepBy` symbol ";")
    case parts of
        []       -> fail "expected multi-affine expression"
        [single] -> pure single
        _        -> fail "expected a single multi-affine expression"

pMultiAffPart :: [SpaceDim] -> Parser MultiAffineExpr
pMultiAffPart params = pBraced <|> pBare
  where
    pBraced = between (symbol "{") (symbol "}") (pMultiAffPartBody params)
    pBare = pMultiAffPartBody params

pMultiAffPartBody :: [SpaceDim] -> Parser MultiAffineExpr
pMultiAffPartBody params = do
    tuple <- pTuple
    let baseSpace = mkSetSpace params tuple
    _ <- symbol "->"
    exprs <- pAffineTuple baseSpace
    let outTuple = Tuple Nothing (replicate (length exprs) (SpaceDim Nothing))
        multiSpace =
            Space
                { spaceParams = params
                , spaceIn = tuple
                , spaceOut = outTuple
                }
    pure $ MultiAffineExpr multiSpace exprs

pAffineTuple :: Space -> Parser [AffineExpr]
pAffineTuple space =
    between (symbol "[") (symbol "]") (pAffineTupleElem space `sepBy` symbol ",")

pAffineTupleElem :: Space -> Parser AffineExpr
pAffineTupleElem space =
    between (symbol "(") (symbol ")") (pAffineExprInlineAff space)
        <|> pAffineExprInlineAff space

pAffineExprInlineAff :: Space -> Parser AffineExpr
pAffineExprInlineAff space = do
    raw <- pLinearExpr (DimEnv space [])
    let ls0 = LocalSpace space []
        (_, affExpr) = convertAffine ls0 raw
    pure affExpr

pMultiPwAffineExpr :: Parser MultiPwAffineExpr
pMultiPwAffineExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <-
        between (symbol "[") (symbol "]") (pMultiPwAffPart (fromMaybe [] params) `sepBy` symbol ";")
    case parts of
        []       -> fail "expected multi-pw-affine expression"
        [single] -> pure single
        _        -> fail "expected a single multi-pw-affine expression"

pMultiUnionPwAffineExpr :: Parser MultiUnionPwAffineExpr
pMultiUnionPwAffineExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <-
        between (symbol "[") (symbol "]") (pMultiPwAffPart (fromMaybe [] params) `sepBy` symbol ";")
    case parts of
        [] -> fail "expected multi-union pw-affine expression"
        _  -> pure $ MultiUnionPwAffineExpr parts

pMultiPwAffPart :: [SpaceDim] -> Parser MultiPwAffineExpr
pMultiPwAffPart params = pBraced <|> pBare
  where
    pBraced = between (symbol "{") (symbol "}") (pMultiPwAffPartBody params)
    pBare = pMultiPwAffPartBody params

pMultiPwAffPartBody :: [SpaceDim] -> Parser MultiPwAffineExpr
pMultiPwAffPartBody params = do
    tuple <- pTuple
    let baseSpace = mkSetSpace params tuple
    _ <- symbol "->"
    exprs <- pPwAffineTuple baseSpace
    let outTuple = Tuple Nothing (replicate (length exprs) (SpaceDim Nothing))
        multiSpace =
            Space
                { spaceParams = params
                , spaceIn = tuple
                , spaceOut = outTuple
                }
    pure $ MultiPwAffineExpr multiSpace exprs

pPwAffineTuple :: Space -> Parser [PwAffineExpr]
pPwAffineTuple space =
    between (symbol "[") (symbol "]") (pPwAffineTupleElem space `sepBy` symbol ",")

pPwAffineTupleElem :: Space -> Parser PwAffineExpr
pPwAffineTupleElem space =
    between (symbol "(") (symbol ")") (pPwAffineExprInline space)
        <|> pPwAffineExprInline space

pPwAffineExprInline :: Space -> Parser PwAffineExpr
pPwAffineExprInline space =
    try (pPwAffineExprMatching space) <|> pPwAffineExprFromLinear space

pPwAffineExprMatching :: Space -> Parser PwAffineExpr
pPwAffineExprMatching space = do
    expr <- pPwAffineExprFullWithParams space.spaceParams
    if expr.pwSpace == space
        then pure expr
        else fail "piecewise affine expression has inconsistent space"

pPwAffineExprFromLinear :: Space -> Parser PwAffineExpr
pPwAffineExprFromLinear space = do
    raw <- pLinearExpr (DimEnv space [])
    let ls0 = LocalSpace space []
        (_, affExpr) = convertAffine ls0 raw
        basicSet = BasicSet (LocalSpace space []) []
    pure $ PwAffineExpr space [(basicSet, affExpr)]

pScheduleTree :: Parser ScheduleTree
pScheduleTree = do
    fields <- between (symbol "{") (symbol "}") (pScheduleField `sepBy` symbol ",")
    buildScheduleTree fields

data ScheduleField
    = FieldDomain UnionSetExpr
    | FieldContext SetExpr
    | FieldFilter UnionSetExpr
    | FieldGuard SetExpr
    | FieldExtension UnionMapExpr
    | FieldExpansion UnionMapExpr
    | FieldSchedule MultiUnionPwAffineExpr
    | FieldSequence [ScheduleTree]
    | FieldSet [ScheduleTree]
    | FieldChild ScheduleTree
    | FieldPermutable Bool
    | FieldCoincident [Bool]
    | FieldAstBuildOptions UnionSetExpr
    | FieldMark Text

pScheduleField :: Parser ScheduleField
pScheduleField =
    pFieldDomain
        <|> pFieldContext
        <|> pFieldFilter
        <|> pFieldGuard
        <|> pFieldExtension
        <|> pFieldExpansion
        <|> pFieldSchedule
        <|> pFieldSequence
        <|> pFieldSet
        <|> pFieldChild
        <|> pFieldPermutable
        <|> pFieldCoincident
        <|> pFieldAstBuildOptions
        <|> pFieldMark

pFieldDomain :: Parser ScheduleField
pFieldDomain = FieldDomain <$> (keyword "domain" *> symbol ":" *> pUnionSetLiteral)

pFieldContext :: Parser ScheduleField
pFieldContext = FieldContext <$> (keyword "context" *> symbol ":" *> pSetLiteral)

pFieldFilter :: Parser ScheduleField
pFieldFilter = FieldFilter <$> (keyword "filter" *> symbol ":" *> pUnionSetLiteral)

pFieldGuard :: Parser ScheduleField
pFieldGuard = FieldGuard <$> (keyword "guard" *> symbol ":" *> pSetLiteral)

pFieldExtension :: Parser ScheduleField
pFieldExtension = FieldExtension <$> (keyword "extension" *> symbol ":" *> pUnionMapLiteral)

pFieldExpansion :: Parser ScheduleField
pFieldExpansion = FieldExpansion <$> (keyword "expansion" *> symbol ":" *> pUnionMapLiteral)

pFieldSchedule :: Parser ScheduleField
pFieldSchedule = FieldSchedule <$> (keyword "schedule" *> symbol ":" *> pMultiUnionPwAffineLiteral)

pFieldSequence :: Parser ScheduleField
pFieldSequence = FieldSequence <$> (keyword "sequence" *> symbol ":" *> pNodeList)

pFieldSet :: Parser ScheduleField
pFieldSet = FieldSet <$> (keyword "set" *> symbol ":" *> pNodeList)

pFieldChild :: Parser ScheduleField
pFieldChild = FieldChild <$> (keyword "child" *> symbol ":" *> pScheduleTree)

pFieldPermutable :: Parser ScheduleField
pFieldPermutable = FieldPermutable <$> (keyword "permutable" *> symbol ":" *> pBool)

pFieldCoincident :: Parser ScheduleField
pFieldCoincident = FieldCoincident <$> (keyword "coincident" *> symbol ":" *> pBoolList)

pFieldAstBuildOptions :: Parser ScheduleField
pFieldAstBuildOptions = FieldAstBuildOptions <$> (keyword "ast_build_options" *> symbol ":" *> pUnionSetLiteral)

pFieldMark :: Parser ScheduleField
pFieldMark = FieldMark <$> (keyword "mark" *> symbol ":" *> pStringLiteral)

pNodeList :: Parser [ScheduleTree]
pNodeList = between (symbol "[") (symbol "]") (pScheduleTree `sepBy` symbol ",")

pBool :: Parser Bool
pBool =
    (symbol "1" $> True)
        <|> (symbol "0" $> False)
        <|> (keyword "true" $> True)
        <|> (keyword "false" $> False)

pBoolList :: Parser [Bool]
pBoolList = between (symbol "[") (symbol "]") (pBool `sepBy` symbol ",")

pStringLiteral :: Parser Text
pStringLiteral = lexeme $ do
    _ <- char '"'
    content <- manyTill L.charLiteral (char '"')
    pure $ T.pack content

pSetLiteral :: Parser SetExpr
pSetLiteral = do
    content <- pStringLiteral
    case parseSetExpr (T.unpack content) of
        Left err     -> fail err
        Right result -> pure result

pUnionSetLiteral :: Parser UnionSetExpr
pUnionSetLiteral = do
    content <- pStringLiteral
    case parseUnionSetExpr (T.unpack content) of
        Left err     -> fail err
        Right result -> pure result

pUnionMapLiteral :: Parser UnionMapExpr
pUnionMapLiteral = do
    content <- pStringLiteral
    case parseUnionMapExpr (T.unpack content) of
        Left err     -> fail err
        Right result -> pure result

pMultiUnionPwAffineLiteral :: Parser MultiUnionPwAffineExpr
pMultiUnionPwAffineLiteral = do
    content <- pStringLiteral
    case parseMultiUnionPwAffineExpr (T.unpack content) of
        Left err     -> fail err
        Right result -> pure result

buildScheduleTree :: [ScheduleField] -> Parser ScheduleTree
buildScheduleTree fields =
    case () of
        _
            | nodeTypeCount > 1 ->
                fail "multiple schedule node types"
            | Just domain <- pickOne isDomain ->
                TreeDomain domain <$> childOrLeaf
            | Just ctx <- pickOne isContext ->
                TreeContext ctx <$> childOrLeaf
            | Just filt <- pickOne isFilter ->
                TreeFilter filt <$> childOrLeaf
            | Just guard <- pickOne isGuard ->
                TreeGuard guard <$> childOrLeaf
            | Just ext <- pickOne isExtension ->
                TreeExtension ext <$> childOrLeaf
            | Just expansion <- pickOne isExpansion ->
                TreeExpansion expansion <$> childOrLeaf
            | Just sched <- pickOne isSchedule ->
                TreeBand (buildBand sched) <$> childOrLeaf
            | Just mark <- pickOne isMark ->
                TreeMark mark <$> childOrLeaf
            | Just seqNodes <- pickOne isSequence ->
                pure $ TreeSequence seqNodes
            | Just setNodes <- pickOne isSet ->
                pure $ TreeSet setNodes
            | null fields ->
                pure TreeLeaf
            | otherwise ->
                fail "invalid schedule tree node"
  where
    pickOne :: (ScheduleField -> Maybe a) -> Maybe a
    pickOne f =
        case [x | field <- fields, Just x <- [f field]] of
            []  -> Nothing
            [x] -> Just x
            _   -> Nothing

    isDomain (FieldDomain x) = Just x
    isDomain _               = Nothing

    isContext (FieldContext x) = Just x
    isContext _                = Nothing

    isFilter (FieldFilter x) = Just x
    isFilter _               = Nothing

    isGuard (FieldGuard x) = Just x
    isGuard _              = Nothing

    isExtension (FieldExtension x) = Just x
    isExtension _                  = Nothing

    isExpansion (FieldExpansion x) = Just x
    isExpansion _                  = Nothing

    isSchedule (FieldSchedule x) = Just x
    isSchedule _                 = Nothing

    isSequence (FieldSequence x) = Just x
    isSequence _                 = Nothing

    isSet (FieldSet x) = Just x
    isSet _            = Nothing

    isMark (FieldMark x) = Just x
    isMark _             = Nothing

    permutable =
        case [x | FieldPermutable x <- fields] of
            (x : _) -> x
            []      -> False

    coincident =
        case [x | FieldCoincident x <- fields] of
            (x : _) -> x
            []      -> []

    astBuildOptions =
        case [x | FieldAstBuildOptions x <- fields] of
            (x : _) -> Just x
            []      -> Nothing

    buildBand sched =
        Band
            { bandSchedule = sched
            , bandPermutable = permutable
            , bandCoincident = coincident
            , bandAstBuildOptions = astBuildOptions
            }

    childOrLeaf =
        case [x | FieldChild x <- fields] of
            []    -> pure [TreeLeaf]
            [one] -> pure [one]
            _     -> fail "multiple child fields"

    nodeTypeCount =
        length
            [ ()
            | field <- fields
            , isNodeType field
            ]

    isNodeType FieldDomain{}    = True
    isNodeType FieldContext{}   = True
    isNodeType FieldFilter{}    = True
    isNodeType FieldGuard{}     = True
    isNodeType FieldExtension{} = True
    isNodeType FieldExpansion{} = True
    isNodeType FieldSchedule{}  = True
    isNodeType FieldSequence{}  = True
    isNodeType FieldSet{}       = True
    isNodeType FieldMark{}      = True
    isNodeType _                = False

data RawLinearExpr = RawLinearExpr
    { rawConstant :: Integer
    , rawCoeffs   :: Map DimRef Integer
    , rawDivTerms :: [RawDivTerm]
    }

newtype RawDivTerm = RawDivTerm (Integer, RawDivExpr)

unpackRawDivTerm :: RawDivTerm -> (Integer, RawDivExpr)
unpackRawDivTerm (RawDivTerm term) = term

data RawDivExpr = RawDivExpr
    { rawDivNumerator   :: RawLinearExpr
    , rawDivDenominator :: Integer
    }

data RawConstraint = RawConstraint Relation RawLinearExpr RawLinearExpr

data DimEnv = DimEnv
    { envSpace  :: Space
    , envLocals :: [LocalDim]
    }

pLinearExpr :: DimEnv -> Parser RawLinearExpr
pLinearExpr env = do
    dimMap <- dimMapOrFail env
    let pVar = do
            name <- pIdent
            case Map.lookup name dimMap of
                Nothing  -> fail $ "unknown dimension: " ++ T.unpack name
                Just ref -> pure ref
        pDiv = pDivExpr env
    first <- pSignedTerm pVar pDiv
    rest <- many ((,) <$> pAddOp <*> pTerm pVar pDiv)
    let terms = first : map applyOp rest
    pure $ rawLinearFromTerms terms
  where
    applyOp (op, term) = op term

pSignedTerm :: Parser DimRef -> Parser RawDivExpr -> Parser RawTerm
pSignedTerm pVar pDiv = do
    sign <- optional (symbol "-" <|> symbol "+")
    term <- pTerm pVar pDiv
    pure $ case sign of
        Just "-" -> negateTerm term
        _        -> term

pAddOp :: Parser (RawTerm -> RawTerm)
pAddOp =
    (symbol "+" $> id)
        <|> (symbol "-" $> negateTerm)

pTerm :: Parser DimRef -> Parser RawDivExpr -> Parser RawTerm
pTerm pVar pDiv =
    try
        ( do
            coeff <- pNumber
            _ <- symbol "*"
            RawTermDiv coeff <$> pDiv
        )
        <|> try
            ( do
                coeff <- pNumber
                _ <- symbol "*"
                RawTermVar coeff <$> pVar
            )
        <|> (RawTermDiv 1 <$> pDiv)
        <|> (RawTermVar 1 <$> pVar)
        <|> (RawTermConst <$> pNumber)

pNumber :: Parser Integer
pNumber = lexeme L.decimal

negateTerm :: RawTerm -> RawTerm
negateTerm (RawTermConst v)    = RawTermConst (-v)
negateTerm (RawTermVar v ref)  = RawTermVar (-v) ref
negateTerm (RawTermDiv v expr) = RawTermDiv (-v) expr

rawLinearFromTerms :: [RawTerm] -> RawLinearExpr
rawLinearFromTerms terms =
    let (cVal, coeffMap, divAcc) = foldl' step (0, Map.empty, []) terms
        filtered = Map.filter (/= 0) coeffMap
        divTerms =
            [ RawDivTerm (coeff, expr)
            | RawDivTerm (coeff, expr) <- reverse divAcc
            , coeff /= 0
            ]
     in RawLinearExpr cVal filtered divTerms
  where
    step (cAcc, mAcc, dAcc) term =
        case term of
            RawTermConst v    -> (cAcc + v, mAcc, dAcc)
            RawTermVar v ref  -> (cAcc, Map.insertWith (+) ref v mAcc, dAcc)
            RawTermDiv v expr -> (cAcc, mAcc, RawDivTerm (v, expr) : dAcc)

pDivExpr :: DimEnv -> Parser RawDivExpr
pDivExpr env = do
    _ <- keyword "floor" <|> keyword "floord"
    _ <- symbol "("
    numerator <-
        try (between (symbol "(") (symbol ")") (pLinearExpr env))
            <|> pLinearExpr env
    _ <- symbol "/"
    denom <- lexeme L.decimal
    _ <- symbol ")"
    pure $ RawDivExpr numerator denom

dimMapOrFail :: DimEnv -> Parser (Map Text DimRef)
dimMapOrFail env =
    let entries =
            paramEntries ++ inEntries ++ outEntries ++ localEntries
        paramEntries =
            [ (name, DimRef ParamDim idx)
            | (idx, SpaceDim (Just name)) <- zip [0 ..] env.envSpace.spaceParams
            ]
        inEntries =
            [ (name, DimRef InDim idx)
            | (idx, SpaceDim (Just name)) <- zip [0 ..] env.envSpace.spaceIn.tupleDims
            ]
        outEntries =
            [ (name, DimRef OutDim idx)
            | (idx, SpaceDim (Just name)) <- zip [0 ..] env.envSpace.spaceOut.tupleDims
            ]
        localEntries =
            [ (name, DimRef LocalDimKind idx)
            | (idx, LocalDim (Just name) _) <- zip [0 ..] env.envLocals
            ]
        dimMap = Map.fromList entries
     in if Map.size dimMap /= length entries
            then fail "duplicate dimension name"
            else pure dimMap

data RawTerm
    = RawTermConst Integer
    | RawTermVar Integer DimRef
    | RawTermDiv Integer RawDivExpr

convertConstraints :: LocalSpace -> [RawConstraint] -> (LocalSpace, [Constraint])
convertConstraints ls0 rawConstraints =
    let (lsFinal, revCons) = foldl' step (ls0, []) rawConstraints
        constraints = reverse revCons
        normalized = map (setConstraintLocalSpace lsFinal) constraints
     in (lsFinal, normalized)
  where
    step (ls, acc) (RawConstraint rel lhs rhs) =
        let (ls', lhsAff) = convertAffine ls lhs
            (ls'', rhsAff) = convertAffine ls' rhs
         in (ls'', Constraint rel lhsAff rhsAff : acc)

convertAffine :: LocalSpace -> RawLinearExpr -> (LocalSpace, AffineExpr)
convertAffine ls raw =
    let (ls', form) = convertLinear ls raw
     in (ls', AffineExpr ls' form)

convertLinear :: LocalSpace -> RawLinearExpr -> (LocalSpace, LinearExpr)
convertLinear ls raw =
    let (ls', coeffs) = foldl' addDiv (ls, raw.rawCoeffs) raw.rawDivTerms
     in (ls', LinearExpr raw.rawConstant coeffs)
  where
    addDiv (lsAcc, coeffMap) term =
        let (coeff, divExpr) = unpackRawDivTerm term
            (ls', ref) = convertDivExpr lsAcc divExpr
            coeffMap' = Map.insertWith (+) ref coeff coeffMap
         in (ls', coeffMap')

convertDivExpr :: LocalSpace -> RawDivExpr -> (LocalSpace, DimRef)
convertDivExpr ls raw =
    let (ls', numer) = convertLinear ls raw.rawDivNumerator
        idx = length ls'.localSpaceDims
        newDim =
            LocalDim
                { localDimName = Nothing
                , localDimDef = Just (DivDef numer raw.rawDivDenominator)
                }
        ls'' = ls'{localSpaceDims = ls'.localSpaceDims ++ [newDim]}
     in (ls'', DimRef LocalDimKind idx)

setConstraintLocalSpace :: LocalSpace -> Constraint -> Constraint
setConstraintLocalSpace ls (Constraint rel lhs rhs) =
    Constraint rel lhs{affLocalSpace = ls} rhs{affLocalSpace = ls}

groupBasicSets :: [BasicSet] -> [SetExpr]
groupBasicSets = foldl' insert []
  where
    insert acc part =
        let base = part.basicSetSpace.localSpaceBase
         in case break ((== base) . (.setSpace)) acc of
                (before, setExpr : after) ->
                    before ++ [setExpr{setParts = setExpr.setParts ++ [part]}] ++ after
                _ -> acc ++ [SetExpr base [part]]

groupBasicMaps :: [BasicMap] -> [MapExpr]
groupBasicMaps = foldl' insert []
  where
    insert acc part =
        let base = part.basicMapSpace.localSpaceBase
         in case break ((== base) . (.mapSpace)) acc of
                (before, mapExpr : after) ->
                    before ++ [mapExpr{mapParts = mapExpr.mapParts ++ [part]}] ++ after
                _ -> acc ++ [MapExpr base [part]]
