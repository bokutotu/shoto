module ISL.Ast.Show (
    setExprToString,
    unionSetExprToString,
    mapExprToString,
    unionMapExprToString,
    affineExprToString,
    multiAffineExprToString,
    scheduleTreeToString,
    constraintToString,
    setExprToCompactString,
    unionSetExprToCompactString,
    mapExprToCompactString,
    unionMapExprToCompactString,
    multiAffineExprToCompactString,
) where

import           Data.List       (dropWhileEnd, intercalate)
import qualified Data.Map.Strict as Map
import           Data.Ratio      (denominator, numerator)
import           Data.Text       (Text)
import qualified Data.Text       as T
import           ISL.Ast.Types

setExprToString :: SetExpr -> String
setExprToString = renderLines . ppSetExpr 0

unionSetExprToString :: UnionSetExpr -> String
unionSetExprToString = renderLines . ppUnionSetExpr 0

mapExprToString :: MapExpr -> String
mapExprToString = renderLines . ppMapExpr 0

unionMapExprToString :: UnionMapExpr -> String
unionMapExprToString = renderLines . ppUnionMapExpr 0

affineExprToString :: AffineExpr -> String
affineExprToString = renderLines . ppAffineExprFull 0

multiAffineExprToString :: MultiAffineExpr -> String
multiAffineExprToString = renderLines . ppMultiAffineExpr 0

scheduleTreeToString :: ScheduleTree -> String
scheduleTreeToString = renderLines . ppScheduleTree 0

constraintToString :: Constraint -> String
constraintToString (Constraint rel lhs rhs) =
    affineExprToString lhs ++ " " ++ ppRelation rel ++ " " ++ affineExprToString rhs

setExprToCompactString :: SetExpr -> String
setExprToCompactString = compactString . setExprToString

unionSetExprToCompactString :: UnionSetExpr -> String
unionSetExprToCompactString = compactString . unionSetExprToString

mapExprToCompactString :: MapExpr -> String
mapExprToCompactString = compactString . mapExprToString

unionMapExprToCompactString :: UnionMapExpr -> String
unionMapExprToCompactString = compactString . unionMapExprToString

multiAffineExprToCompactString :: MultiAffineExpr -> String
multiAffineExprToCompactString = compactString . multiAffineExprToString

renderLines :: [String] -> String
renderLines = intercalate "\n"

indent :: Int -> String
indent n = replicate n ' '

compactString :: String -> String
compactString = unwords . map strip . lines
  where
    strip = dropWhile (== ' ') . dropWhileEnd (== ' ')

ppSetExpr :: Int -> SetExpr -> [String]
ppSetExpr base setExpr =
    let params = spaceParams (setSpace setExpr)
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = ppSetPart (base + 2) setExpr
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppUnionSetExpr :: Int -> UnionSetExpr -> [String]
ppUnionSetExpr base (UnionSetExpr sets) =
    let params = commonParamsSet sets
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = joinPartsWithSemicolons (map (ppSetPart (base + 2)) sets)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppMapExpr :: Int -> MapExpr -> [String]
ppMapExpr base mapExpr =
    let params = commonParamsMap [mapExpr]
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = ppMapPart (base + 2) mapExpr
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppUnionMapExpr :: Int -> UnionMapExpr -> [String]
ppUnionMapExpr base (UnionMapExpr maps) =
    let params = commonParamsMap maps
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = joinPartsWithSemicolons (map (ppMapPart (base + 2)) maps)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppSetPart :: Int -> SetExpr -> [String]
ppSetPart base (SetExpr space constraints) =
    let tupleLine =
            indent base
                ++ ppTuple (spaceName space) (spaceInputs space)
                ++ if null constraints then "" else " :"
        constraintLines = ppConstraintsLines (base + 2) constraints
     in tupleLine : constraintLines

ppMapPart :: Int -> MapExpr -> [String]
ppMapPart base (MapExpr dom ran constraints) =
    let domDims = spaceTupleInputs dom
        ranDims = spaceTupleOutputs ran
        tupleLine =
            indent base
                ++ ppTuple (spaceName dom) domDims
                ++ " -> "
                ++ ppTuple (spaceName ran) ranDims
                ++ if null constraints then "" else " :"
        constraintLines = ppConstraintsLines (base + 2) constraints
     in tupleLine : constraintLines

ppConstraintsLines :: Int -> [Constraint] -> [String]
ppConstraintsLines _ [] = []
ppConstraintsLines base constraints =
    let constraintStrs = addAnd (map ppConstraintInline constraints)
     in map (indent base ++) constraintStrs

ppConstraintInline :: Constraint -> String
ppConstraintInline (Constraint rel lhs rhs) =
    ppAffineExprInline lhs ++ " " ++ ppRelation rel ++ " " ++ ppAffineExprInline rhs

ppAffineExprInline :: AffineExpr -> String
ppAffineExprInline (AffineLinear lin) = ppLinearExpr lin
ppAffineExprInline aff =
    compactString (affineExprToString aff)

ppAffineExprFull :: Int -> AffineExpr -> [String]
ppAffineExprFull base (AffineLinear lin) =
    let space = linearSpace lin
        params = spaceParams space
        tuple = ppTuple (spaceName space) (spaceInputs space)
        line = indent base ++ ppParamsPrefix params ++ "{ " ++ tuple ++ " -> " ++ ppLinearExpr lin ++ " }"
     in [line]
ppAffineExprFull base (AffinePiecewise space parts) =
    let params = spaceParams space
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = joinPartsWithSemicolons (map (ppAffinePiece (base + 2)) parts)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppAffinePiece :: Int -> (SetExpr, LinearExpr) -> [String]
ppAffinePiece base (setExpr, lin) =
    let space = setSpace setExpr
        tupleLine =
            indent base
                ++ ppTuple (spaceName space) (spaceInputs space)
                ++ " -> "
                ++ ppLinearExpr lin
                ++ if null (setConstraints setExpr) then "" else " :"
        constraintLines = ppConstraintsLines (base + 2) (setConstraints setExpr)
     in tupleLine : constraintLines

ppLinearExpr :: LinearExpr -> String
ppLinearExpr (LinearExpr space constantVal coeffMap) =
    let orderedRefs = dimRefsInOrder space
        orderedTerms =
            [ termFromCoeff ref coeff
            | ref <- orderedRefs
            , let coeff = Map.findWithDefault 0 ref coeffMap
            , coeff /= 0
            ]
        extraTerms =
            [ termFromCoeff ref coeff
            | (ref, coeff) <- Map.toList coeffMap
            , ref `notElem` orderedRefs
            , coeff /= 0
            ]
        constTerms =
            [ termFromConst constantVal
            | not (constantVal == 0 && (not (null orderedTerms) || not (null extraTerms)))
            ]
        terms = orderedTerms ++ extraTerms ++ constTerms
     in renderSignedTerms terms

ppRelation :: Relation -> String
ppRelation RelEq = "="
ppRelation RelLe = "<="
ppRelation RelGe = ">="

ppParamsPrefix :: [SpaceDim] -> String
ppParamsPrefix []   = ""
ppParamsPrefix dims = ppDimList dims ++ " -> "

ppTuple :: Maybe Text -> [SpaceDim] -> String
ppTuple name dims =
    maybe "" T.unpack name ++ ppDimList dims

ppDimList :: [SpaceDim] -> String
ppDimList dims = "[" ++ intercalate ", " (map ppSpaceDim dims) ++ "]"

ppSpaceDim :: SpaceDim -> String
ppSpaceDim (SpaceDim name) = T.unpack name

spaceTupleInputs :: Space -> [SpaceDim]
spaceTupleInputs space =
    let inputs = spaceInputs space
        outputs = spaceOutputs space
     in if null inputs then outputs else inputs

spaceTupleOutputs :: Space -> [SpaceDim]
spaceTupleOutputs space =
    let inputs = spaceInputs space
        outputs = spaceOutputs space
     in if null outputs then inputs else outputs

dimRefsInOrder :: Space -> [DimRef]
dimRefsInOrder space =
    map (DimRef ParamDim) (spaceParams space)
        ++ map (DimRef InDim) (spaceInputs space)
        ++ map (DimRef OutDim) (spaceOutputs space)

data SignedTerm = SignedTerm Bool String

termFromCoeff :: DimRef -> Rational -> SignedTerm
termFromCoeff ref coeff =
    let neg = coeff < 0
        absCoeff = abs coeff
        name = ppSpaceDim (dimTarget ref)
        body =
            if absCoeff == 1
                then name
                else showRational absCoeff ++ "*" ++ name
     in SignedTerm neg body

termFromConst :: Rational -> SignedTerm
termFromConst coeff =
    let neg = coeff < 0
        body = showRational (abs coeff)
     in SignedTerm neg body

renderSignedTerms :: [SignedTerm] -> String
renderSignedTerms [] = "0"
renderSignedTerms (SignedTerm neg body : rest) =
    let first = (if neg then "-" else "") ++ body
        tailParts =
            [ (if nextNeg then " - " else " + ") ++ nextBody
            | SignedTerm nextNeg nextBody <- rest
            ]
     in first ++ concat tailParts

showRational :: Rational -> String
showRational r =
    let n = numerator r
        d = denominator r
     in if d == 1 then show n else show n ++ "/" ++ show d

addAnd :: [String] -> [String]
addAnd []       = []
addAnd [x]      = [x]
addAnd (x : xs) = (x ++ " and") : addAnd xs

joinPartsWithSemicolons :: [[String]] -> [String]
joinPartsWithSemicolons [] = []
joinPartsWithSemicolons [part] = part
joinPartsWithSemicolons (part : rest) =
    appendSemicolon part ++ joinPartsWithSemicolons rest

appendSemicolon :: [String] -> [String]
appendSemicolon [] = []
appendSemicolon linesList =
    let body = init linesList
        lastLine = last linesList
     in body ++ [lastLine ++ ";"]

commonParamsSet :: [SetExpr] -> [SpaceDim]
commonParamsSet [] = []
commonParamsSet (SetExpr space _ : rest) =
    let params = spaceParams space
     in if all ((== params) . spaceParams . setSpace) rest
            then params
            else error "UnionSetExpr has inconsistent parameters"

commonParamsMap :: [MapExpr] -> [SpaceDim]
commonParamsMap [] = []
commonParamsMap (MapExpr dom ran _ : rest) =
    let params = spaceParams dom
     in if params == spaceParams ran && all (sameParams params) rest
            then params
            else error "UnionMapExpr has inconsistent parameters"
  where
    sameParams params (MapExpr dom' ran' _) =
        spaceParams dom' == params && spaceParams ran' == params

multiParts :: MultiAffineExpr -> [(Space, [AffineExpr])]
multiParts (MultiAffineExpr space exprs) = [(space, exprs)]
multiParts (MultiAffineUnion parts)      = parts

ppMultiAffineExpr :: Int -> MultiAffineExpr -> [String]
ppMultiAffineExpr base expr =
    let parts = multiParts expr
        params = commonParamsMulti parts
        header = indent base ++ ppParamsPrefix params ++ "["
        body = joinPartsWithSemicolons (map (ppMultiPart (base + 2)) parts)
        footer = indent base ++ "]"
     in header : body ++ [footer]

ppMultiPart :: Int -> (Space, [AffineExpr]) -> [String]
ppMultiPart base (space, exprs) =
    let tupleLine =
            indent base
                ++ "{ "
                ++ ppTuple (spaceName space) (spaceInputs space)
                ++ " -> "
                ++ ppAffineTuple exprs
                ++ " }"
     in [tupleLine]

ppAffineTuple :: [AffineExpr] -> String
ppAffineTuple exprs =
    "[" ++ intercalate ", " (map ppAffineTupleElem exprs) ++ "]"

ppAffineTupleElem :: AffineExpr -> String
ppAffineTupleElem expr =
    "(" ++ ppAffineExprInline expr ++ ")"

commonParamsMulti :: [(Space, [AffineExpr])] -> [SpaceDim]
commonParamsMulti [] = []
commonParamsMulti ((space, _) : rest) =
    let params = spaceParams space
     in if all ((== params) . spaceParams . fst) rest
            then params
            else error "MultiAffineExpr has inconsistent parameters"

ppScheduleTree :: Int -> ScheduleTree -> [String]
ppScheduleTree base tree =
    renderNode base (scheduleFields tree)
  where
    scheduleFields :: ScheduleTree -> [(String, NodeValue)]
    scheduleFields node =
        case node of
            TreeDomain uset children ->
                fieldWithChild "domain" (NodeScalar (quote (unionSetExprToCompactString uset))) children
            TreeContext setExpr children ->
                fieldWithChild "context" (NodeScalar (quote (setExprToCompactString setExpr))) children
            TreeFilter uset children ->
                fieldWithChild "filter" (NodeScalar (quote (unionSetExprToCompactString uset))) children
            TreeGuard setExpr children ->
                fieldWithChild "guard" (NodeScalar (quote (setExprToCompactString setExpr))) children
            TreeExtension umap children ->
                fieldWithChild "extension" (NodeScalar (quote (unionMapExprToCompactString umap))) children
            TreeBand bandExpr perm children ->
                let baseFields =
                        [ ("schedule", NodeScalar (quote (multiAffineExprToCompactString bandExpr)))
                        , ("permutable", NodeScalar (if perm then "1" else "0"))
                        ]
                 in baseFields ++ childField children
            TreeMark name children ->
                fieldWithChild "mark" (NodeScalar (quote (T.unpack name))) children
            TreeSequence children ->
                [("sequence", NodeList children)]
            TreeSet children ->
                [("set", NodeList children)]
            TreeLeaf ->
                []

data NodeValue
    = NodeScalar String
    | NodeTree ScheduleTree
    | NodeList [ScheduleTree]

fieldWithChild :: String -> NodeValue -> [ScheduleTree] -> [(String, NodeValue)]
fieldWithChild name value children =
    (name, value) : childField children

childField :: [ScheduleTree] -> [(String, NodeValue)]
childField children =
    case dropLeaves children of
        []      -> []
        [child] -> [("child", NodeTree child)]
        _       -> error "ScheduleTree node has multiple children"

dropLeaves :: [ScheduleTree] -> [ScheduleTree]
dropLeaves = filter (not . isLeaf)

isLeaf :: ScheduleTree -> Bool
isLeaf TreeLeaf = True
isLeaf _        = False

renderNode :: Int -> [(String, NodeValue)] -> [String]
renderNode base fields =
    let open = indent base ++ "{"
        close = indent base ++ "}"
        fieldLines = renderFields (base + 2) fields
     in open : fieldLines ++ [close]

renderFields :: Int -> [(String, NodeValue)] -> [String]
renderFields _ [] = []
renderFields base fields =
    concat $
        zipWith
            (\idx field -> renderField base (idx == length fields - 1) field)
            [0 ..]
            fields

renderField :: Int -> Bool -> (String, NodeValue) -> [String]
renderField base isLast (name, value) =
    let prefix = indent base ++ name ++ ": "
        valueLines =
            case value of
                NodeScalar str ->
                    [prefix ++ str]
                NodeTree node ->
                    inlineFirstLine prefix (ppScheduleTree (base + 2) node)
                NodeList nodes ->
                    inlineFirstLine prefix (ppNodeList (base + 2) nodes)
        finalLines =
            if isLast
                then valueLines
                else appendCommaToLast valueLines
     in finalLines

ppNodeList :: Int -> [ScheduleTree] -> [String]
ppNodeList base nodes =
    let open = indent base ++ "["
        close = indent base ++ "]"
        itemLines = renderListItems (base + 2) nodes
     in open : itemLines ++ [close]

renderListItems :: Int -> [ScheduleTree] -> [String]
renderListItems _ [] = []
renderListItems base nodes =
    concat $
        zipWith
            (\idx node -> renderListItem base (idx == length nodes - 1) node)
            [0 ..]
            nodes

renderListItem :: Int -> Bool -> ScheduleTree -> [String]
renderListItem base isLast node =
    let linesList = ppScheduleTree base node
        finalLines =
            if isLast
                then linesList
                else appendCommaToLast linesList
     in finalLines

inlineFirstLine :: String -> [String] -> [String]
inlineFirstLine prefix [] = [prefix]
inlineFirstLine prefix (first : rest) =
    let stripped = dropWhile (== ' ') first
     in (prefix ++ stripped) : rest

appendCommaToLast :: [String] -> [String]
appendCommaToLast [] = []
appendCommaToLast xs =
    let body = init xs
        lastLine = last xs
     in body ++ [lastLine ++ ","]

quote :: String -> String
quote str = "\"" ++ concatMap escapeChar str ++ "\""

escapeChar :: Char -> String
escapeChar '"'  = "\\\""
escapeChar '\\' = "\\\\"
escapeChar '\n' = "\\n"
escapeChar c    = [c]
