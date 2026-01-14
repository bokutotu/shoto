module ISL.Ast.Read (
    parseSetExpr,
    parseUnionSetExpr,
    parseMapExpr,
    parseUnionMapExpr,
    parseAffineExpr,
    parseMultiAffineExpr,
    parseScheduleTree,
    parseConstraint,
) where

import           Control.Applicative        (empty, (<|>))
import           Data.Functor               (($>), (<&>))
import           Data.Map.Strict            (Map)
import qualified Data.Map.Strict            as Map
import           Data.Maybe                 (fromMaybe)
import           Data.Ratio                 ((%))
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

parseMultiAffineExpr :: String -> Either String MultiAffineExpr
parseMultiAffineExpr = parseWith pMultiAffineExpr

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
pDimList = between (symbol "[") (symbol "]") (pIdent `sepBy` symbol ",") <&> map SpaceDim

pTuple :: Parser (Maybe Text, [SpaceDim])
pTuple = try namedTuple <|> unnamedTuple
  where
    namedTuple = do
        name <- pIdent
        dims <- pDimList
        pure (Just name, dims)
    unnamedTuple = do
        dims <- pDimList
        pure (Nothing, dims)

pSetExpr :: Parser SetExpr
pSetExpr = do
    UnionSetExpr parts <- pUnionSetExpr
    case parts of
        [single] -> pure single
        _        -> fail "expected a single set"

pUnionSetExpr :: Parser UnionSetExpr
pUnionSetExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <- between (symbol "{") (symbol "}") (pSetPart (fromMaybe [] params) `sepBy` symbol ";")
    pure $ UnionSetExpr parts

pSetPart :: [SpaceDim] -> Parser SetExpr
pSetPart params = do
    (name, dims) <- pTuple
    let space =
            Space
                { spaceName = name
                , spaceParams = params
                , spaceInputs = dims
                , spaceOutputs = []
                }
    constraints <- optional (symbol ":" *> pConstraints space)
    pure $ SetExpr space (fromMaybe [] constraints)

pMapExpr :: Parser MapExpr
pMapExpr = do
    UnionMapExpr parts <- pUnionMapExpr
    case parts of
        [single] -> pure single
        _        -> fail "expected a single map"

pUnionMapExpr :: Parser UnionMapExpr
pUnionMapExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <- between (symbol "{") (symbol "}") (pMapPart (fromMaybe [] params) `sepBy` symbol ";")
    pure $ UnionMapExpr parts

pMapPart :: [SpaceDim] -> Parser MapExpr
pMapPart params = do
    (domName, domDims) <- pTuple
    _ <- symbol "->"
    (ranName, ranDims) <- pTuple
    let domSpace =
            Space
                { spaceName = domName
                , spaceParams = params
                , spaceInputs = domDims
                , spaceOutputs = []
                }
        ranSpace =
            Space
                { spaceName = ranName
                , spaceParams = params
                , spaceInputs = []
                , spaceOutputs = ranDims
                }
        mapSpace =
            Space
                { spaceName = Nothing
                , spaceParams = params
                , spaceInputs = domDims
                , spaceOutputs = ranDims
                }
    constraints <- optional (symbol ":" *> pConstraints mapSpace)
    pure $ MapExpr domSpace ranSpace (fromMaybe [] constraints)

pConstraints :: Space -> Parser [Constraint]
pConstraints space = concat <$> pConstraintChain space `sepBy1` keyword "and"

pConstraintChain :: Space -> Parser [Constraint]
pConstraintChain space = do
    first <- pAffineExprInline space
    rest <- some ((,) <$> pRelOp <*> pAffineExprInline space)
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

buildConstraint :: RelOp -> AffineExpr -> AffineExpr -> Parser Constraint
buildConstraint op lhs rhs =
    case op of
        OpLe -> pure $ Constraint RelLe lhs rhs
        OpGe -> pure $ Constraint RelGe lhs rhs
        OpEq -> pure $ Constraint RelEq lhs rhs
        OpLt -> Constraint RelLe lhs <$> shiftAffine (-1) rhs
        OpGt -> Constraint RelGe lhs <$> shiftAffine 1 rhs

shiftAffine :: Rational -> AffineExpr -> Parser AffineExpr
shiftAffine delta (AffineLinear lin) =
    pure $ AffineLinear lin{constant = constant lin + delta}
shiftAffine _ (AffinePiecewise _ _) =
    fail "strict inequality requires linear expression"

pAffineExprInline :: Space -> Parser AffineExpr
pAffineExprInline space = AffineLinear <$> pLinearExpr space

pAffineExprFull :: Parser AffineExpr
pAffineExprFull = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <- between (symbol "{") (symbol "}") (pAffinePiece (fromMaybe [] params) `sepBy` symbol ";")
    case parts of
        [] -> fail "expected affine expression"
        [single@(setExpr, lin)] ->
            if null (setConstraints setExpr)
                then pure $ AffineLinear lin
                else do
                    space <- commonPieceSpace parts
                    pure $ AffinePiecewise space [single]
        _ -> do
            space <- commonPieceSpace parts
            pure $ AffinePiecewise space parts

pAffinePiece :: [SpaceDim] -> Parser (SetExpr, LinearExpr)
pAffinePiece params = do
    (name, dims) <- pTuple
    let space =
            Space
                { spaceName = name
                , spaceParams = params
                , spaceInputs = dims
                , spaceOutputs = []
                }
    _ <- symbol "->"
    lin <- pLinearExpr space
    constraints <- optional (symbol ":" *> pConstraints space)
    let setExpr = SetExpr space (fromMaybe [] constraints)
    pure (setExpr, lin)

commonPieceSpace :: [(SetExpr, LinearExpr)] -> Parser Space
commonPieceSpace [] = fail "expected affine pieces"
commonPieceSpace ((setExpr, _) : rest) =
    let baseSpace = setSpace setExpr
        allSame = all ((== baseSpace) . setSpace . fst) rest
     in if allSame
            then pure baseSpace
            else fail "piecewise affine parts have inconsistent spaces"

pConstraintFull :: Parser Constraint
pConstraintFull = do
    lhs <- pAffineExprFull
    op <- pRelOp
    rhs <- pAffineExprFull
    buildConstraint op lhs rhs

pMultiAffineExpr :: Parser MultiAffineExpr
pMultiAffineExpr = do
    params <- optional (try (pDimList <* symbol "->"))
    parts <- between (symbol "[") (symbol "]") (pMultiPart (fromMaybe [] params) `sepBy` symbol ";")
    case parts of
        []       -> fail "expected multi-affine expression"
        [single] -> pure $ uncurry MultiAffineExpr single
        _        -> pure $ MultiAffineUnion parts

pMultiPart :: [SpaceDim] -> Parser (Space, [AffineExpr])
pMultiPart params = pBraced <|> pBare
  where
    pBraced = between (symbol "{") (symbol "}") (pMultiPartBody params)
    pBare = pMultiPartBody params

pMultiPartBody :: [SpaceDim] -> Parser (Space, [AffineExpr])
pMultiPartBody params = do
    (name, dims) <- pTuple
    let space =
            Space
                { spaceName = name
                , spaceParams = params
                , spaceInputs = dims
                , spaceOutputs = []
                }
    _ <- symbol "->"
    exprs <- pAffineTuple space
    pure (space, exprs)

pAffineTuple :: Space -> Parser [AffineExpr]
pAffineTuple space =
    between (symbol "[") (symbol "]") (pAffineTupleElem space `sepBy` symbol ",")

pAffineTupleElem :: Space -> Parser AffineExpr
pAffineTupleElem space =
    between (symbol "(") (symbol ")") (pAffineExprInline space)
        <|> pAffineExprInline space

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
    | FieldSchedule MultiAffineExpr
    | FieldSequence [ScheduleTree]
    | FieldSet [ScheduleTree]
    | FieldChild ScheduleTree
    | FieldPermutable Bool
    | FieldMark Text

pScheduleField :: Parser ScheduleField
pScheduleField =
    pFieldDomain
        <|> pFieldContext
        <|> pFieldFilter
        <|> pFieldGuard
        <|> pFieldExtension
        <|> pFieldSchedule
        <|> pFieldSequence
        <|> pFieldSet
        <|> pFieldChild
        <|> pFieldPermutable
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

pFieldSchedule :: Parser ScheduleField
pFieldSchedule = FieldSchedule <$> (keyword "schedule" *> symbol ":" *> pMultiAffineLiteral)

pFieldSequence :: Parser ScheduleField
pFieldSequence = FieldSequence <$> (keyword "sequence" *> symbol ":" *> pNodeList)

pFieldSet :: Parser ScheduleField
pFieldSet = FieldSet <$> (keyword "set" *> symbol ":" *> pNodeList)

pFieldChild :: Parser ScheduleField
pFieldChild = FieldChild <$> (keyword "child" *> symbol ":" *> pScheduleTree)

pFieldPermutable :: Parser ScheduleField
pFieldPermutable = FieldPermutable <$> (keyword "permutable" *> symbol ":" *> pBool)

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

pStringLiteral :: Parser Text
pStringLiteral = do
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

pMultiAffineLiteral :: Parser MultiAffineExpr
pMultiAffineLiteral = do
    content <- pStringLiteral
    case parseMultiAffineExpr (T.unpack content) of
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
            | Just sched <- pickOne isSchedule ->
                TreeBand sched permutable <$> childOrLeaf
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
    isNodeType FieldSchedule{}  = True
    isNodeType FieldSequence{}  = True
    isNodeType FieldSet{}       = True
    isNodeType FieldMark{}      = True
    isNodeType _                = False

data Term
    = TermConst Rational
    | TermVar Rational DimRef

pLinearExpr :: Space -> Parser LinearExpr
pLinearExpr space = do
    dimMap <- dimMapOrFail space
    let pVar = do
            name <- pIdent
            case Map.lookup name dimMap of
                Nothing  -> fail $ "unknown dimension: " ++ T.unpack name
                Just ref -> pure ref
    first <- pSignedTerm pVar
    rest <- many ((,) <$> pAddOp <*> pTerm pVar)
    let terms = first : map applyOp rest
    pure $ linearFromTerms space terms
  where
    applyOp (op, term) = op term

pSignedTerm :: Parser DimRef -> Parser Term
pSignedTerm pVar = do
    sign <- optional (symbol "-" <|> symbol "+")
    term <- pTerm pVar
    pure $ case sign of
        Just "-" -> negateTerm term
        _        -> term

pAddOp :: Parser (Term -> Term)
pAddOp =
    (symbol "+" $> id)
        <|> (symbol "-" $> negateTerm)

pTerm :: Parser DimRef -> Parser Term
pTerm pVar =
    try
        ( do
            coeff <- pNumber
            _ <- symbol "*"
            TermVar coeff <$> pVar
        )
        <|> (TermVar 1 <$> pVar)
        <|> (TermConst <$> pNumber)

pNumber :: Parser Rational
pNumber = lexeme $ do
    num <- L.decimal
    denom <- optional (char '/' *> L.decimal)
    pure $ case denom of
        Nothing -> fromInteger num
        Just d  -> num % d

negateTerm :: Term -> Term
negateTerm (TermConst v)   = TermConst (-v)
negateTerm (TermVar v ref) = TermVar (-v) ref

linearFromTerms :: Space -> [Term] -> LinearExpr
linearFromTerms space terms =
    let (cVal, coeffMap) = foldl step (0, Map.empty) terms
        filtered = Map.filter (/= 0) coeffMap
     in LinearExpr space cVal filtered
  where
    step (cAcc, mAcc) term =
        case term of
            TermConst v   -> (cAcc + v, mAcc)
            TermVar v ref -> (cAcc, Map.insertWith (+) ref v mAcc)

dimMapOrFail :: Space -> Parser (Map Text DimRef)
dimMapOrFail space =
    let entries =
            [(spaceDimName d, DimRef ParamDim d) | d <- spaceParams space]
                ++ [(spaceDimName d, DimRef InDim d) | d <- spaceInputs space]
                ++ [(spaceDimName d, DimRef OutDim d) | d <- spaceOutputs space]
        dimMap = Map.fromList entries
     in if Map.size dimMap /= length entries
            then fail "duplicate dimension name"
            else pure dimMap
