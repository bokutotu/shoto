{-# LANGUAGE OverloadedRecordDot #-}

module ISL.Ast.Show (
    setExprToString,
    unionSetExprToString,
    mapExprToString,
    unionMapExprToString,
    affineExprToString,
    pwAffineExprToString,
    multiAffineExprToString,
    multiPwAffineExprToString,
    multiUnionPwAffineExprToString,
    scheduleTreeToString,
    constraintToString,
    setExprToCompactString,
    unionSetExprToCompactString,
    mapExprToCompactString,
    unionMapExprToCompactString,
    pwAffineExprToCompactString,
    multiAffineExprToCompactString,
    multiPwAffineExprToCompactString,
    multiUnionPwAffineExprToCompactString,
) where

import           Data.List       (dropWhileEnd, intercalate)
import qualified Data.Map.Strict as Map
import           Data.Maybe      (isNothing)
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

pwAffineExprToString :: PwAffineExpr -> String
pwAffineExprToString = renderLines . ppPwAffineExprFull 0

multiAffineExprToString :: MultiAffineExpr -> String
multiAffineExprToString = renderLines . ppMultiAffineExpr 0

multiPwAffineExprToString :: MultiPwAffineExpr -> String
multiPwAffineExprToString = renderLines . ppMultiPwAffineExpr 0

multiUnionPwAffineExprToString :: MultiUnionPwAffineExpr -> String
multiUnionPwAffineExprToString = renderLines . ppMultiUnionPwAffineExpr 0

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

pwAffineExprToCompactString :: PwAffineExpr -> String
pwAffineExprToCompactString = compactString . pwAffineExprToString

multiAffineExprToCompactString :: MultiAffineExpr -> String
multiAffineExprToCompactString = compactString . multiAffineExprToString

multiPwAffineExprToCompactString :: MultiPwAffineExpr -> String
multiPwAffineExprToCompactString = compactString . multiPwAffineExprToString

multiUnionPwAffineExprToCompactString :: MultiUnionPwAffineExpr -> String
multiUnionPwAffineExprToCompactString = compactString . multiUnionPwAffineExprToString

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
    let params = setExpr.setSpace.spaceParams
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = joinPartsWithSemicolons (map (ppBasicSetPart (base + 2)) setExpr.setParts)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppUnionSetExpr :: Int -> UnionSetExpr -> [String]
ppUnionSetExpr base (UnionSetExpr sets) =
    let params = commonParamsSet sets
        header = indent base ++ ppParamsPrefix params ++ "{"
        body =
            joinPartsWithSemicolons
                (concatMap (map (ppBasicSetPart (base + 2)) . (.setParts)) sets)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppMapExpr :: Int -> MapExpr -> [String]
ppMapExpr base mapExpr =
    let params = commonParamsMap [mapExpr]
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = joinPartsWithSemicolons (map (ppBasicMapPart (base + 2) mapExpr.mapSpace) mapExpr.mapParts)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppUnionMapExpr :: Int -> UnionMapExpr -> [String]
ppUnionMapExpr base (UnionMapExpr maps) =
    let params = commonParamsMap maps
        header = indent base ++ ppParamsPrefix params ++ "{"
        body =
            joinPartsWithSemicolons
                (concatMap (\mapExpr -> map (ppBasicMapPart (base + 2) mapExpr.mapSpace) mapExpr.mapParts) maps)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppBasicSetPart :: Int -> BasicSet -> [String]
ppBasicSetPart base (BasicSet localSpace constraints) =
    let space = localSpace.localSpaceBase
        tupleLine =
            indent base
                ++ ppTupleIn space
                ++ if null constraints then "" else " :"
        constraintLines = ppConstraintsLines (base + 2) localSpace constraints
     in tupleLine : constraintLines

ppBasicMapPart :: Int -> Space -> BasicMap -> [String]
ppBasicMapPart base baseSpace (BasicMap localSpace constraints) =
    let tupleLine =
            indent base
                ++ ppTupleIn baseSpace
                ++ " -> "
                ++ ppTupleOut baseSpace
                ++ if null constraints then "" else " :"
        constraintLines = ppConstraintsLines (base + 2) localSpace constraints
     in tupleLine : constraintLines

ppConstraintsLines :: Int -> LocalSpace -> [Constraint] -> [String]
ppConstraintsLines _ _ [] = []
ppConstraintsLines base localSpace constraints =
    let localNames =
            [ localDimDisplayName idx dim
            | (idx, dim) <- zip [0 ..] localSpace.localSpaceDims
            , isNothing dim.localDimDef
            ]
        constraintStrs = addAnd (map ppConstraintInline constraints)
        wrapped = wrapExists localNames constraintStrs
     in map (indent base ++) wrapped

ppConstraintInline :: Constraint -> String
ppConstraintInline (Constraint rel lhs rhs) =
    ppAffineExprInline lhs ++ " " ++ ppRelation rel ++ " " ++ ppAffineExprInline rhs

ppAffineExprInline :: AffineExpr -> String
ppAffineExprInline aff =
    let ls = aff.affLocalSpace
     in ppLinearExpr ls aff.affForm

ppAffineExprFull :: Int -> AffineExpr -> [String]
ppAffineExprFull base aff =
    let localSpace = aff.affLocalSpace
        space = localSpace.localSpaceBase
        params = space.spaceParams
        tuple = ppTupleIn space
        line =
            indent base
                ++ ppParamsPrefix params
                ++ "{ "
                ++ tuple
                ++ " -> "
                ++ ppLinearExpr localSpace aff.affForm
                ++ " }"
     in [line]

ppPwAffineExprFull :: Int -> PwAffineExpr -> [String]
ppPwAffineExprFull base pw =
    let params = pw.pwSpace.spaceParams
        header = indent base ++ ppParamsPrefix params ++ "{"
        body = joinPartsWithSemicolons (map (ppPwAffinePiece (base + 2)) pw.pwPieces)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppPwAffineExprFullNoParams :: Int -> PwAffineExpr -> [String]
ppPwAffineExprFullNoParams base pw =
    let header = indent base ++ "{"
        body = joinPartsWithSemicolons (map (ppPwAffinePiece (base + 2)) pw.pwPieces)
        footer = indent base ++ "}"
     in header : body ++ [footer]

ppPwAffinePiece :: Int -> (BasicSet, AffineExpr) -> [String]
ppPwAffinePiece base (basicSet, aff) =
    let space = basicSet.basicSetSpace.localSpaceBase
        tupleLine =
            indent base
                ++ ppTupleIn space
                ++ " -> "
                ++ ppLinearExpr aff.affLocalSpace aff.affForm
                ++ if null basicSet.basicSetConstraints then "" else " :"
        constraintLines = ppConstraintsLines (base + 2) basicSet.basicSetSpace basicSet.basicSetConstraints
     in tupleLine : constraintLines

ppRelation :: Relation -> String
ppRelation RelEq = "="
ppRelation RelLe = "<="
ppRelation RelGe = ">="

ppParamsPrefix :: [SpaceDim] -> String
ppParamsPrefix []   = ""
ppParamsPrefix dims = ppDimListWith (spaceDimNameOr "p") dims ++ " -> "

ppTupleIn :: Space -> String
ppTupleIn space = ppTupleWith (spaceDimNameOr "i") space.spaceIn

ppTupleOut :: Space -> String
ppTupleOut space = ppTupleWith (spaceDimNameOr "o") space.spaceOut

ppTupleWith :: (Int -> SpaceDim -> String) -> Tuple -> String
ppTupleWith nameForDim tuple =
    maybe "" T.unpack tuple.tupleName ++ ppDimListWith nameForDim tuple.tupleDims

ppDimListWith :: (Int -> SpaceDim -> String) -> [SpaceDim] -> String
ppDimListWith nameForDim dims =
    "[" ++ intercalate ", " [nameForDim idx dim | (idx, dim) <- zip [0 ..] dims] ++ "]"

spaceDimNameOr :: String -> Int -> SpaceDim -> String
spaceDimNameOr _ _ (SpaceDim (Just name))    = T.unpack name
spaceDimNameOr prefix idx (SpaceDim Nothing) = prefix ++ show idx

localDimDisplayName :: Int -> LocalDim -> String
localDimDisplayName idx dim = maybe ("t" ++ show idx) T.unpack dim.localDimName

ppLinearExpr :: LocalSpace -> LinearExpr -> String
ppLinearExpr localSpace expr =
    let orderedRefs = dimRefsInOrder localSpace
        orderedTerms =
            [ termFromCoeff localSpace ref coeff
            | ref <- orderedRefs
            , let coeff = Map.findWithDefault 0 ref expr.linearCoeffs
            , coeff /= 0
            ]
        extraTerms =
            [ termFromCoeff localSpace ref coeff
            | (ref, coeff) <- Map.toList expr.linearCoeffs
            , ref `notElem` orderedRefs
            , coeff /= 0
            ]
        constTerms =
            [ termFromConst expr.linearConstant
            | not
                ( expr.linearConstant == 0
                    && ( not (null orderedTerms)
                            || not (null extraTerms)
                       )
                )
            ]
        terms = orderedTerms ++ extraTerms ++ constTerms
     in renderSignedTerms terms

dimRefsInOrder :: LocalSpace -> [DimRef]
dimRefsInOrder localSpace =
    let space = localSpace.localSpaceBase
        params = space.spaceParams
        inputs = space.spaceIn.tupleDims
        outputs = space.spaceOut.tupleDims
        locals = localSpace.localSpaceDims
        paramRefs = [DimRef ParamDim idx | idx <- [0 .. length params - 1]]
        inputRefs = [DimRef InDim idx | idx <- [0 .. length inputs - 1]]
        outputRefs = [DimRef OutDim idx | idx <- [0 .. length outputs - 1]]
        localRefs = [DimRef LocalDimKind idx | idx <- [0 .. length locals - 1]]
     in paramRefs ++ inputRefs ++ outputRefs ++ localRefs

data SignedTerm = SignedTerm Bool String

termFromCoeff :: LocalSpace -> DimRef -> Integer -> SignedTerm
termFromCoeff localSpace ref coeff =
    let neg = coeff < 0
        absCoeff = abs coeff
        bodyExpr = ppDimRef localSpace ref
        body =
            if absCoeff == 1
                then bodyExpr
                else show absCoeff ++ "*" ++ bodyExpr
     in SignedTerm neg body

ppDimRef :: LocalSpace -> DimRef -> String
ppDimRef localSpace ref =
    let space = localSpace.localSpaceBase
     in case ref.dimKind of
            ParamDim ->
                let dim = space.spaceParams !! ref.dimPos
                 in spaceDimNameOr "p" ref.dimPos dim
            InDim ->
                let dim = space.spaceIn.tupleDims !! ref.dimPos
                 in spaceDimNameOr "i" ref.dimPos dim
            OutDim ->
                let dim = space.spaceOut.tupleDims !! ref.dimPos
                 in spaceDimNameOr "o" ref.dimPos dim
            LocalDimKind ->
                let dim = localSpace.localSpaceDims !! ref.dimPos
                 in case dim.localDimDef of
                        Nothing  -> localDimDisplayName ref.dimPos dim
                        Just def -> ppDivDef localSpace def

termFromConst :: Integer -> SignedTerm
termFromConst coeff =
    let neg = coeff < 0
        body = show (abs coeff)
     in SignedTerm neg body

ppDivDef :: LocalSpace -> DivDef -> String
ppDivDef localSpace (DivDef numer denom) =
    "floor((" ++ ppLinearExpr localSpace numer ++ ")/" ++ show denom ++ ")"

renderSignedTerms :: [SignedTerm] -> String
renderSignedTerms [] = "0"
renderSignedTerms (SignedTerm neg body : rest) =
    let first = (if neg then "-" else "") ++ body
        tailParts =
            [ (if nextNeg then " - " else " + ") ++ nextBody
            | SignedTerm nextNeg nextBody <- rest
            ]
     in first ++ concat tailParts

addAnd :: [String] -> [String]
addAnd []       = []
addAnd [x]      = [x]
addAnd (x : xs) = (x ++ " and") : addAnd xs

wrapExists :: [String] -> [String] -> [String]
wrapExists [] linesList = linesList
wrapExists _ [] = []
wrapExists locals [single] =
    ["exists (" ++ intercalate ", " locals ++ ": " ++ single ++ ")"]
wrapExists locals (firstLine : rest) =
    let prefix = "exists (" ++ intercalate ", " locals ++ ": "
        firstWithPrefix = prefix ++ firstLine
     in case reverse rest of
            [] -> [firstWithPrefix ++ ")"]
            lastLine : revMiddle -> firstWithPrefix : reverse revMiddle ++ [lastLine ++ ")"]

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
    let params = space.spaceParams
     in if all ((== params) . (.spaceParams) . (.setSpace)) rest
            then params
            else error "UnionSetExpr has inconsistent parameters"

commonParamsMap :: [MapExpr] -> [SpaceDim]
commonParamsMap [] = []
commonParamsMap (MapExpr space _ : rest) =
    let params = space.spaceParams
     in if all ((== params) . (.spaceParams) . (.mapSpace)) rest
            then params
            else error "UnionMapExpr has inconsistent parameters"

ppMultiAffineExpr :: Int -> MultiAffineExpr -> [String]
ppMultiAffineExpr base (MultiAffineExpr space exprs) =
    let params = space.spaceParams
        header = indent base ++ ppParamsPrefix params ++ "["
        body =
            [ indent (base + 2)
                ++ "{ "
                ++ ppTupleIn space
                ++ " -> "
                ++ ppAffineTuple exprs
                ++ " }"
            ]
        footer = indent base ++ "]"
     in header : body ++ [footer]

ppMultiPwAffineExpr :: Int -> MultiPwAffineExpr -> [String]
ppMultiPwAffineExpr base expr =
    let params = expr.multiPwAffSpace.spaceParams
        header = indent base ++ ppParamsPrefix params ++ "["
        body = ppMultiPwAffinePart (base + 2) expr
        footer = indent base ++ "]"
     in header : body ++ [footer]

ppMultiUnionPwAffineExpr :: Int -> MultiUnionPwAffineExpr -> [String]
ppMultiUnionPwAffineExpr base (MultiUnionPwAffineExpr parts) =
    let params = commonParamsMulti parts
        header = indent base ++ ppParamsPrefix params ++ "["
        body = joinPartsWithSemicolons (map (ppMultiPwAffinePart (base + 2)) parts)
        footer = indent base ++ "]"
     in header : body ++ [footer]

ppMultiPwAffinePart :: Int -> MultiPwAffineExpr -> [String]
ppMultiPwAffinePart base (MultiPwAffineExpr space exprs) =
    let tupleLine =
            indent base
                ++ "{ "
                ++ ppTupleIn space
                ++ " -> "
                ++ ppPwAffineTuple exprs
                ++ " }"
     in [tupleLine]

ppPwAffineTuple :: [PwAffineExpr] -> String
ppPwAffineTuple exprs =
    "[" ++ intercalate ", " (map ppPwAffineTupleElem exprs) ++ "]"

ppPwAffineTupleElem :: PwAffineExpr -> String
ppPwAffineTupleElem expr =
    "(" ++ ppPwAffineExprInline expr ++ ")"

ppPwAffineExprInline :: PwAffineExpr -> String
ppPwAffineExprInline expr =
    case expr.pwPieces of
        [(basicSet, aff)]
            | null basicSet.basicSetConstraints ->
                ppLinearExpr aff.affLocalSpace aff.affForm
        _ -> compactString (renderLines (ppPwAffineExprFullNoParams 0 expr))

ppAffineTuple :: [AffineExpr] -> String
ppAffineTuple exprs =
    "[" ++ intercalate ", " (map ppAffineTupleElem exprs) ++ "]"

ppAffineTupleElem :: AffineExpr -> String
ppAffineTupleElem expr =
    "(" ++ ppAffineExprInline expr ++ ")"

commonParamsMulti :: [MultiPwAffineExpr] -> [SpaceDim]
commonParamsMulti [] = []
commonParamsMulti (MultiPwAffineExpr space _ : rest) =
    let params = space.spaceParams
     in if all ((== params) . (.spaceParams) . (.multiPwAffSpace)) rest
            then params
            else error "MultiUnionPwAffineExpr has inconsistent parameters"

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
            TreeExpansion umap children ->
                fieldWithChild "expansion" (NodeScalar (quote (unionMapExprToCompactString umap))) children
            TreeBand band children ->
                let baseFields =
                        [ ("schedule", NodeScalar (quote (multiUnionPwAffineExprToCompactString band.bandSchedule)))
                        , ("permutable", NodeScalar (if band.bandPermutable then "1" else "0"))
                        ]
                    coincidentField =
                        [ ("coincident", NodeScalar (ppBoolList band.bandCoincident))
                        | not (null band.bandCoincident)
                        ]
                    optionsField =
                        case band.bandAstBuildOptions of
                            Nothing -> []
                            Just opts ->
                                [("ast_build_options", NodeScalar (quote (unionSetExprToCompactString opts)))]
                 in baseFields ++ coincidentField ++ optionsField ++ childField children
            TreeMark name children ->
                fieldWithChild "mark" (NodeScalar (quote (T.unpack name))) children
            TreeSequence children ->
                [("sequence", NodeList children)]
            TreeSet children ->
                [("set", NodeList children)]
            TreeLeaf ->
                []

ppBoolList :: [Bool] -> String
ppBoolList values =
    "[" ++ intercalate ", " [if v then "1" else "0" | v <- values] ++ "]"

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
escapeChar c    = [c]
