{-# LANGUAGE OverloadedRecordDot #-}

module FrontendIR.Lowering (
    CheckedProgram,
    checkProgram,
    lowerToRaw,
) where

import           Control.Monad      (forM_, unless)
import           Data.List          (intercalate)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map.Strict    as Map
import qualified Data.Set           as Set
import           FrontendIR.Types   (Axis (..), Expr (..), FrontendError (..),
                                     IterName, IxExpr (..), ParamName,
                                     Program (..), Stmt (..), TensorDecl (..),
                                     TensorName, iterNameToString,
                                     paramNameToString, tensorNameToString)
import           Polyhedral.Parse   (RawPolyhedralModel (..))

data CheckedProgram = CheckedProgram
    { checkedParams :: [ParamName]
    , checkedIters :: [IterName]
    , checkedIterExtents :: Map.Map IterName ParamName
    , checkedStmts :: [CheckedStmt]
    }

data CheckedStmt
    = CAssign
        { cOutputTensor :: TensorName
        , cOutputIndex :: [IterName]
        , cLoads :: [(TensorName, [IterName])]
        }
    | CReduction
        { cOutputTensor :: TensorName
        , cOutputIndex :: [IterName]
        , cLoads :: [(TensorName, [IterName])]
        , cReductionAxes :: [IterName]
        }

data ScheduleExpr
    = ScheduleIter IterName
    | ScheduleConst Int
    deriving (Eq, Ord, Show)

data LoweredStmt = LoweredStmt
    { loweredName :: String
    , loweredDomainIters :: [IterName]
    , loweredOrderTuple :: [ScheduleExpr]
    , loweredLoads :: [(TensorName, [IterName])]
    , loweredOutputTensor :: TensorName
    , loweredOutputIndex :: [IterName]
    , loweredIsReduction :: Bool
    }

lowerToRaw :: CheckedProgram -> RawPolyhedralModel
lowerToRaw prog =
    case prog.checkedStmts of
        [singleStmt] ->
            buildRaw prog.checkedParams prog.checkedIterExtents [lowerSingleStmt prog.checkedIters singleStmt]
        multipleStmts ->
            buildRaw
                prog.checkedParams
                prog.checkedIterExtents
                (lowerMultiStmts prog.checkedIters multipleStmts)

checkProgram :: Program -> Either FrontendError CheckedProgram
checkProgram Program{axes = axisList, tensors = tensorDecls, stmts = statementList} = do
    maybe (Right ()) (Left . ErrDuplicateIter) (firstDuplicate ((.iter) <$> axesList))
    maybe (Right ()) (Left . ErrDuplicateParam) (firstDuplicate expectedParams)
    maybe (Right ()) (Left . ErrDuplicateTensor) (firstDuplicate ((.tensor) <$> tensorList))
    forM_ tensorList $ \decl ->
        forM_ decl.shape $ \param ->
            unless (Set.member param knownShapeParams) $
                Left $
                    ErrUnknownTensorShapeParam decl.tensor param
    checkedStatements <-
        case statements of
            [_] -> traverse checkSingleStmt statements
            _ -> checkMultiStmtProgram statements
    pure
        CheckedProgram
            { checkedParams = expectedParams
            , checkedIters = expectedIters
            , checkedIterExtents = iterExtents
            , checkedStmts = checkedStatements
            }
  where
    axesList = NE.toList axisList
    tensorList = NE.toList tensorDecls
    statements = NE.toList statementList
    expectedIters = (.iter) <$> axesList
    expectedParams = (.extent) <$> axesList
    knownIterNames = Set.fromList expectedIters
    knownShapeParams = Set.fromList expectedParams
    iterExtents = Map.fromList [(axis.iter, axis.extent) | axis <- axesList]
    tensorRanks = Map.fromList [(decl.tensor, length decl.shape) | decl <- tensorList]

    ensureRankMatches :: TensorName -> [IxExpr] -> Either FrontendError ()
    ensureRankMatches tensorName indices =
        case Map.lookup tensorName tensorRanks of
            Nothing -> Left $ ErrUndeclaredTensor tensorName
            Just expectedRank ->
                unless (length indices == expectedRank) $
                    Left $
                        ErrTensorRankMismatch tensorName expectedRank (length indices)

    ensureKnownIndexIters :: [IterName] -> Either FrontendError ()
    ensureKnownIndexIters indexIters =
        forM_ indexIters $ \iter ->
            unless (Set.member iter knownIterNames) $
                Left $
                    ErrUnknownIndexIter iter

    checkSingleStmt :: Stmt -> Either FrontendError CheckedStmt
    checkSingleStmt currentStmt =
        case currentStmt of
            Assign{} -> do
                let actualStore = ixExprName <$> currentStmt.outputIndex
                unless (actualStore == expectedIters) $
                    Left $
                        ErrStoreIndexMismatch expectedIters actualStore
                ensureRankMatches currentStmt.outputTensor currentStmt.outputIndex
                readAccesses <- checkAssignExpr expectedIters currentStmt.rhs
                pure
                    CAssign
                        { cOutputTensor = currentStmt.outputTensor
                        , cOutputIndex = actualStore
                        , cLoads = readAccesses
                        }
            Reduction{} -> do
                let actualStore = ixExprName <$> currentStmt.outputIndex
                let reducedAxes = reducedAxesFrom actualStore expectedIters
                ensureKnownIndexIters actualStore
                unless (isOrderedSubsequence actualStore expectedIters) $
                    Left $
                        ErrReductionOutputNotSubsequence expectedIters actualStore
                unless (not (null reducedAxes)) $
                    Left ErrReductionRequiresReducedAxis
                ensureRankMatches currentStmt.outputTensor currentStmt.outputIndex
                readAccesses <- checkReductionExpr currentStmt.rhs
                pure
                    CReduction
                        { cOutputTensor = currentStmt.outputTensor
                        , cOutputIndex = actualStore
                        , cLoads = readAccesses
                        , cReductionAxes = reducedAxes
                        }

    checkMultiStmtProgram :: [Stmt] -> Either FrontendError [CheckedStmt]
    checkMultiStmtProgram [] = pure []
    checkMultiStmtProgram (firstStmt : restStmts) = do
        let spatialIters = ixExprName <$> firstStmt.outputIndex
        ensureKnownIndexIters spatialIters
        unless (isOrderedSubsequence spatialIters expectedIters) $
            Left $
                ErrStoreIndexMismatch expectedIters spatialIters
        firstChecked <- checkMultiStmt spatialIters firstStmt
        restChecked <- traverse (checkMultiStmt spatialIters) restStmts
        pure (firstChecked : restChecked)

    checkMultiStmt :: [IterName] -> Stmt -> Either FrontendError CheckedStmt
    checkMultiStmt spatialIters currentStmt = do
        let actualStore = ixExprName <$> currentStmt.outputIndex
        unless (actualStore == spatialIters) $
            Left $
                ErrStoreIndexMismatch spatialIters actualStore
        ensureRankMatches currentStmt.outputTensor currentStmt.outputIndex
        case currentStmt of
            Assign{} -> do
                readAccesses <- checkAssignExpr spatialIters currentStmt.rhs
                pure
                    CAssign
                        { cOutputTensor = currentStmt.outputTensor
                        , cOutputIndex = actualStore
                        , cLoads = readAccesses
                        }
            Reduction{} -> do
                let reducedAxes = reducedAxesFrom spatialIters expectedIters
                unless (not (null reducedAxes)) $
                    Left ErrReductionRequiresReducedAxis
                unless (length reducedAxes == 1) $
                    Left $
                        ErrMultiStmtRequiresSingleReductionAxis reducedAxes
                readAccesses <- checkReductionExpr currentStmt.rhs
                pure
                    CReduction
                        { cOutputTensor = currentStmt.outputTensor
                        , cOutputIndex = actualStore
                        , cLoads = readAccesses
                        , cReductionAxes = reducedAxes
                        }

    checkAssignExpr :: [IterName] -> Expr -> Either FrontendError [(TensorName, [IterName])]
    checkAssignExpr _ (EConst _) = pure []
    checkAssignExpr expectedIndices (EAdd lhs rhs) = (<>) <$> checkAssignExpr expectedIndices lhs <*> checkAssignExpr expectedIndices rhs
    checkAssignExpr expectedIndices (EMul lhs rhs) = (<>) <$> checkAssignExpr expectedIndices lhs <*> checkAssignExpr expectedIndices rhs
    checkAssignExpr expectedIndices (ELoad tensorName indices) = do
        ensureRankMatches tensorName indices
        let actual = ixExprName <$> indices
        unless (actual == expectedIndices) $
            Left $
                ErrLoadIndexMismatch tensorName expectedIndices actual
        pure [(tensorName, actual)]

    checkReductionExpr :: Expr -> Either FrontendError [(TensorName, [IterName])]
    checkReductionExpr (EConst _) = pure []
    checkReductionExpr (EAdd lhs rhs) = (<>) <$> checkReductionExpr lhs <*> checkReductionExpr rhs
    checkReductionExpr (EMul lhs rhs) = (<>) <$> checkReductionExpr lhs <*> checkReductionExpr rhs
    checkReductionExpr (ELoad tensorName indices) = do
        ensureRankMatches tensorName indices
        let actual = ixExprName <$> indices
        ensureKnownIndexIters actual
        pure [(tensorName, actual)]

lowerSingleStmt :: [IterName] -> CheckedStmt -> LoweredStmt
lowerSingleStmt allIters stmt =
    let domainIters = allIters
     in LoweredStmt
            { loweredName = "S"
            , loweredDomainIters = domainIters
            , loweredOrderTuple = mkOrderTuple 0 allIters domainIters
            , loweredLoads = stmt.cLoads
            , loweredOutputTensor = stmt.cOutputTensor
            , loweredOutputIndex = stmt.cOutputIndex
            , loweredIsReduction = isReductionStmt stmt
            }

lowerMultiStmts :: [IterName] -> [CheckedStmt] -> [LoweredStmt]
lowerMultiStmts allIters stmts = zipWith lowerStmt [0 ..] (zip (statementNames (length stmts)) stmts)
  where
    lowerStmt :: Int -> (String, CheckedStmt) -> LoweredStmt
    lowerStmt stage (stmtName, stmt) =
        let domainIters =
                case stmt of
                    CAssign{} -> stmt.cOutputIndex
                    CReduction{} -> stmt.cOutputIndex <> stmt.cReductionAxes
         in LoweredStmt
                { loweredName = stmtName
                , loweredDomainIters = domainIters
                , loweredOrderTuple = mkOrderTuple stage allIters domainIters
                , loweredLoads = stmt.cLoads
                , loweredOutputTensor = stmt.cOutputTensor
                , loweredOutputIndex = stmt.cOutputIndex
                , loweredIsReduction = isReductionStmt stmt
                }

buildRaw :: [ParamName] -> Map.Map IterName ParamName -> [LoweredStmt] -> RawPolyhedralModel
buildRaw params iterExtents loweredStmts =
    let domains =
            [(stmt.loweredName, stmt.loweredDomainIters) | stmt <- loweredStmts]
        orders =
            [(stmt.loweredName, stmt.loweredDomainIters, stmt.loweredOrderTuple) | stmt <- loweredStmts]
        readRelations =
            concatMap mkReadRelations loweredStmts
        writeRelations =
            mkWriteRelation <$> loweredStmts
        reductionStmts = filter (.loweredIsReduction) loweredStmts
        reductionDomains =
            [(stmt.loweredName, stmt.loweredDomainIters) | stmt <- reductionStmts]
        reductionReadRelations = mkReductionRelation <$> reductionStmts
        reductionWriteRelations = mkReductionRelation <$> reductionStmts
     in RawPolyhedralModel
            { context = mkContext params
            , domain = mkDomainUnion params iterExtents domains
            , programOrder = mkProgramOrderUnion params orders
            , readAccess = mkAccessUnion params readRelations
            , writeAccess = mkAccessUnion params writeRelations
            , reductionDomain = mkDomainUnion params iterExtents reductionDomains
            , reductionRead = mkAccessUnion params reductionReadRelations
            , reductionWrite = mkAccessUnion params reductionWriteRelations
            }

mkReadRelations :: LoweredStmt -> [String]
mkReadRelations stmt =
    [ statementRef stmt.loweredName stmt.loweredDomainIters
        <> " -> "
        <> tensorRef tensorName indices
    | (tensorName, indices) <- stmt.loweredLoads
    ]

mkWriteRelation :: LoweredStmt -> String
mkWriteRelation stmt =
    statementRef stmt.loweredName stmt.loweredDomainIters
        <> " -> "
        <> tensorRef stmt.loweredOutputTensor stmt.loweredOutputIndex

mkReductionRelation :: LoweredStmt -> String
mkReductionRelation stmt =
    statementRef stmt.loweredName stmt.loweredDomainIters
        <> " -> "
        <> tensorRef stmt.loweredOutputTensor stmt.loweredOutputIndex

mkContext :: [ParamName] -> String
mkContext params
    | null params = "{ : }"
    | otherwise = withParams params $ "{ : " <> constraints <> " }"
  where
    constraints = intercalate " and " (mkNonNegative <$> params)

mkDomainUnion :: [ParamName] -> Map.Map IterName ParamName -> [(String, [IterName])] -> String
mkDomainUnion _ _ [] = "{ }"
mkDomainUnion params iterExtents domains =
    withOptionalParams params $ "{ " <> intercalate "; " (mkDomainPart <$> domains) <> " }"
  where
    mkDomainPart :: (String, [IterName]) -> String
    mkDomainPart (stmtName, iters)
        | null iters = statementRef stmtName iters
        | otherwise =
            statementRef stmtName iters
                <> " : "
                <> intercalate " and " (mkRange <$> ((\iter -> (iter, lookupIterExtent iter iterExtents)) <$> iters))

mkProgramOrderUnion :: [ParamName] -> [(String, [IterName], [ScheduleExpr])] -> String
mkProgramOrderUnion _ [] = "{ }"
mkProgramOrderUnion params orders =
    withOptionalParams params $ "{ " <> intercalate "; " (mkOrderPart <$> orders) <> " }"
  where
    mkOrderPart :: (String, [IterName], [ScheduleExpr]) -> String
    mkOrderPart (stmtName, domainIters, orderTuple) =
        statementRef stmtName domainIters
            <> " -> "
            <> tuple renderScheduleExpr orderTuple

mkAccessUnion :: [ParamName] -> [String] -> String
mkAccessUnion _ [] = "{ }"
mkAccessUnion params relations =
    withOptionalParams params $ "{ " <> intercalate "; " (uniqueStable relations) <> " }"

withOptionalParams :: [ParamName] -> String -> String
withOptionalParams [] body = body
withOptionalParams params body = withParams params body

withParams :: [ParamName] -> String -> String
withParams params body = tuple paramNameToString params <> " -> " <> body

tuple :: (a -> String) -> [a] -> String
tuple render xs = "[" <> intercalate "," (render <$> xs) <> "]"

statementRef :: String -> [IterName] -> String
statementRef name iters = name <> tuple iterNameToString iters

tensorRef :: TensorName -> [IterName] -> String
tensorRef tensor indices = tensorNameToString tensor <> tuple iterNameToString indices

ixExprName :: IxExpr -> IterName
ixExprName (IxVar name) = name

mkNonNegative :: ParamName -> String
mkNonNegative param = "0 <= " <> paramNameToString param

mkRange :: (IterName, ParamName) -> String
mkRange (iter, extent) =
    "0 <= " <> iterNameToString iter <> " < " <> paramNameToString extent

uniqueStable :: (Ord a) => [a] -> [a]
uniqueStable = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
        | Set.member x seen = go seen xs
        | otherwise = x : go (Set.insert x seen) xs

firstDuplicate :: (Ord a) => [a] -> Maybe a
firstDuplicate = go Set.empty
  where
    go _ [] = Nothing
    go seen (x : xs)
        | Set.member x seen = Just x
        | otherwise = go (Set.insert x seen) xs

isOrderedSubsequence :: (Eq a) => [a] -> [a] -> Bool
isOrderedSubsequence [] _ = True
isOrderedSubsequence _ [] = False
isOrderedSubsequence (x : xs) (y : ys)
    | x == y = isOrderedSubsequence xs ys
    | otherwise = isOrderedSubsequence (x : xs) ys

reducedAxesFrom :: [IterName] -> [IterName] -> [IterName]
reducedAxesFrom outputIters expectedIters =
    let outputSet = Set.fromList outputIters
     in filter (`Set.notMember` outputSet) expectedIters

lookupIterExtent :: IterName -> Map.Map IterName ParamName -> ParamName
lookupIterExtent iter iterExtents =
    case Map.lookup iter iterExtents of
        Just extent -> extent
        Nothing ->
            error $ "missing extent for iterator: " <> iterNameToString iter

statementNames :: Int -> [String]
statementNames count = ["S" <> show idx | idx <- [0 .. count - 1]]

isReductionStmt :: CheckedStmt -> Bool
isReductionStmt CAssign{} = False
isReductionStmt CReduction{} = True

mkOrderTuple :: Int -> [IterName] -> [IterName] -> [ScheduleExpr]
mkOrderTuple stage allIters domainIters =
    ScheduleConst stage : (mkAxisExpr <$> allIters)
  where
    domainIterSet = Set.fromList domainIters

    mkAxisExpr iter
        | Set.member iter domainIterSet = ScheduleIter iter
        | otherwise = ScheduleConst 0

renderScheduleExpr :: ScheduleExpr -> String
renderScheduleExpr expr =
    case expr of
        ScheduleIter iter -> iterNameToString iter
        ScheduleConst value -> show value
