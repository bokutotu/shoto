{-# LANGUAGE OverloadedRecordDot #-}

module FrontendIR.Lowering.RawBuilder (
    lowerToRaw,
) where

import           Data.List                   (intercalate)
import qualified Data.Map.Strict             as Map
import qualified Data.Set                    as Set
import           FrontendIR.Lowering.Checked (CheckedProgram (..),
                                              CheckedStmt (..))
import           FrontendIR.Types            (IterName, ParamName, TensorName,
                                              iterNameToString,
                                              paramNameToString,
                                              tensorNameToString)
import           Polyhedral.Parse            (RawPolyhedralModel (..))

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
    buildRaw
        prog.checkedParams
        prog.checkedIterExtents
        (lowerStmt prog.checkedIters prog.checkedStmts)

lowerStmt :: [IterName] -> [CheckedStmt] -> [LoweredStmt]
lowerStmt allIters stmts = zipWith lowerOne [0 ..] (zip (statementNames (length stmts)) stmts)
  where
    lowerOne :: Int -> (String, CheckedStmt) -> LoweredStmt
    lowerOne stage (stmtName, stmt) =
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
