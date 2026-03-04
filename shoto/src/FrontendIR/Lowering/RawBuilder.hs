{-# LANGUAGE OverloadedRecordDot #-}

module FrontendIR.Lowering.RawBuilder (
    lowerToRaw,
) where

import           Data.List                        (intercalate)
import qualified Data.Map.Strict                  as Map
import qualified Data.Set                         as Set
import           FrontendIR.Lowering.PolyhedralIR
import           FrontendIR.Types                 (IterName, ParamName,
                                                   TensorName, iterNameToString,
                                                   paramNameToString,
                                                   tensorNameToString)
import           Polyhedral.Parse                 (RawPolyhedralModel (..))

lowerToRaw :: PolyProgram -> RawPolyhedralModel
lowerToRaw prog =
    let stmts = prog.pStmts
        redStmts = filter stmtIsReduction stmts
     in RawPolyhedralModel
            { context = mkContext prog.pParams
            , domain = mkDomainUnion prog.pParams prog.pIterExtents (map stmtDomain stmts)
            , programOrder = mkProgramOrderUnion prog.pParams (map stmtSchedule stmts)
            , readAccess = mkAccessUnion prog.pParams (concatMap stmtReads stmts)
            , writeAccess = mkAccessUnion prog.pParams (map stmtWrite stmts)
            , reductionDomain = mkDomainUnion prog.pParams prog.pIterExtents (map stmtDomain redStmts)
            , reductionRead = mkAccessUnion prog.pParams (map stmtWrite redStmts)
            , reductionWrite = mkAccessUnion prog.pParams (map stmtWrite redStmts)
            }

mkContext :: [ParamName] -> String
mkContext params
    | null params = "{ : }"
    | otherwise = withParams params $ "{ : " <> intercalate " and " (mkNonNegative <$> params) <> " }"
  where
    mkNonNegative param = "0 <= " <> paramNameToString param

mkDomainUnion :: [ParamName] -> Map.Map IterName ParamName -> [PolyDomain] -> String
mkDomainUnion _ _ [] = "{ }"
mkDomainUnion params iterExtents domains =
    withOptionalParams params $ "{ " <> intercalate "; " (mkDomainPart <$> domains) <> " }"
  where
    mkDomainPart dom
        | null dom.dIters = statementRef dom.dStmtName dom.dIters
        | otherwise =
            statementRef dom.dStmtName dom.dIters
                <> " : "
                <> intercalate " and " (mkRange iterExtents <$> dom.dIters)

    mkRange extents iter =
        let extent = extents Map.! iter
         in "0 <= " <> iterNameToString iter <> " < " <> paramNameToString extent

mkProgramOrderUnion :: [ParamName] -> [PolySchedule] -> String
mkProgramOrderUnion _ [] = "{ }"
mkProgramOrderUnion params schedules =
    withOptionalParams params $ "{ " <> intercalate "; " (mkOrderPart <$> schedules) <> " }"
  where
    mkOrderPart sched =
        statementRef sched.sStmtName sched.sIters
            <> " -> "
            <> tuple renderScheduleAxis sched.sAxes

mkAccessUnion :: [ParamName] -> [PolyAccess] -> String
mkAccessUnion _ [] = "{ }"
mkAccessUnion params accesses =
    withOptionalParams params $
        "{ " <> intercalate "; " (uniqueStable (mkAccessPart <$> accesses)) <> " }"
  where
    mkAccessPart acc =
        statementRef acc.aStmtName acc.aIters
            <> " -> "
            <> tensorRef acc.aTensor acc.aIndices

withOptionalParams :: [ParamName] -> String -> String
withOptionalParams [] body = body
withOptionalParams params body = withParams params body

withParams :: [ParamName] -> String -> String
withParams params body = tuple paramNameToString params <> " -> " <> body

tuple :: (a -> String) -> [a] -> String
tuple render xs = "[" <> intercalate "," (render <$> xs) <> "]"

statementRef :: StmtName -> [IterName] -> String
statementRef sname iters = stmtNameToString sname <> tuple iterNameToString iters

tensorRef :: TensorName -> [IterName] -> String
tensorRef tensor indices = tensorNameToString tensor <> tuple iterNameToString indices

renderScheduleAxis :: ScheduleAxis -> String
renderScheduleAxis (PadAxis v) = show v
renderScheduleAxis (IterAxis i) = iterNameToString i

uniqueStable :: (Ord a) => [a] -> [a]
uniqueStable = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
        | Set.member x seen = go seen xs
        | otherwise = x : go (Set.insert x seen) xs
