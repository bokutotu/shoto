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
    , checkedStmt :: CheckedStmt
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
        }

lowerToRaw :: CheckedProgram -> RawPolyhedralModel
lowerToRaw prog =
    let writeAccess =
            mkWriteAccess
                prog.checkedParams
                prog.checkedIters
                (prog.checkedStmt.cOutputTensor, prog.checkedStmt.cOutputIndex)
        (reductionDomain, reductionRead, reductionWrite) =
            case prog.checkedStmt of
                CAssign{} ->
                    ("{ }", "{ }", "{ }")
                CReduction{} ->
                    (mkDomain prog.checkedParams prog.checkedIters, writeAccess, writeAccess)
     in RawPolyhedralModel
            { context = mkContext prog.checkedParams
            , domain = mkDomain prog.checkedParams prog.checkedIters
            , programOrder = mkProgramOrder prog.checkedParams prog.checkedIters
            , readAccess = mkReadAccess prog.checkedParams prog.checkedIters prog.checkedStmt.cLoads
            , writeAccess = writeAccess
            , reductionDomain = reductionDomain
            , reductionRead = reductionRead
            , reductionWrite = reductionWrite
            }

checkProgram :: Program -> Either FrontendError CheckedProgram
checkProgram Program{axes = axisList, tensors = tensorDecls, stmt = statement} = do
    maybe (Right ()) (Left . ErrDuplicateIter) (firstDuplicate ((.iter) <$> axesList))
    maybe (Right ()) (Left . ErrDuplicateParam) (firstDuplicate expectedParams)
    maybe (Right ()) (Left . ErrDuplicateTensor) (firstDuplicate ((.tensor) <$> tensorList))
    forM_ tensorList $ \decl ->
        forM_ decl.shape $ \param ->
            unless (Set.member param knownShapeParams) $
                Left $
                    ErrUnknownTensorShapeParam decl.tensor param
    checkedStatement <- checkStmt statement
    pure
        CheckedProgram
            { checkedParams = expectedParams
            , checkedIters = expectedIters
            , checkedStmt = checkedStatement
            }
  where
    axesList = NE.toList axisList
    tensorList = NE.toList tensorDecls
    expectedIters = (.iter) <$> axesList
    expectedParams = (.extent) <$> axesList
    knownIterNames = Set.fromList expectedIters
    knownShapeParams = Set.fromList expectedParams
    tensorRanks = Map.fromList [(decl.tensor, length decl.shape) | decl <- tensorList]

    ensureRankMatches :: TensorName -> [IxExpr] -> Either FrontendError ()
    ensureRankMatches tensorName indices =
        case Map.lookup tensorName tensorRanks of
            Nothing -> Left $ ErrUndeclaredTensor tensorName
            Just expectedRank ->
                unless (length indices == expectedRank) $
                    Left $
                        ErrTensorRankMismatch tensorName expectedRank (length indices)

    checkStmt :: Stmt -> Either FrontendError CheckedStmt
    checkStmt currentStmt =
        case currentStmt of
            Assign{} -> do
                let actualStore = ixExprName <$> currentStmt.outputIndex
                unless (actualStore == expectedIters) $
                    Left $
                        ErrStoreIndexMismatch expectedIters actualStore
                ensureRankMatches currentStmt.outputTensor currentStmt.outputIndex
                readAccesses <- checkAssignExpr currentStmt.rhs
                pure
                    CAssign
                        { cOutputTensor = currentStmt.outputTensor
                        , cOutputIndex = actualStore
                        , cLoads = readAccesses
                        }
            Reduction{} -> do
                let actualStore = ixExprName <$> currentStmt.outputIndex
                forM_ actualStore $ \iter ->
                    unless (Set.member iter knownIterNames) $
                        Left $
                            ErrUnknownIndexIter iter
                unless (isOrderedSubsequence actualStore expectedIters) $
                    Left $
                        ErrReductionOutputNotSubsequence expectedIters actualStore
                unless (length actualStore < length expectedIters) $
                    Left ErrReductionRequiresReducedAxis
                ensureRankMatches currentStmt.outputTensor currentStmt.outputIndex
                readAccesses <- checkReductionExpr currentStmt.rhs
                pure
                    CReduction
                        { cOutputTensor = currentStmt.outputTensor
                        , cOutputIndex = actualStore
                        , cLoads = readAccesses
                        }

    checkAssignExpr :: Expr -> Either FrontendError [(TensorName, [IterName])]
    checkAssignExpr (EConst _) = pure []
    checkAssignExpr (EAdd lhs rhs) = (<>) <$> checkAssignExpr lhs <*> checkAssignExpr rhs
    checkAssignExpr (EMul lhs rhs) = (<>) <$> checkAssignExpr lhs <*> checkAssignExpr rhs
    checkAssignExpr (ELoad tensorName indices) = do
        ensureRankMatches tensorName indices
        let actual = ixExprName <$> indices
        unless (actual == expectedIters) $
            Left $
                ErrLoadIndexMismatch tensorName expectedIters actual
        pure [(tensorName, actual)]

    checkReductionExpr :: Expr -> Either FrontendError [(TensorName, [IterName])]
    checkReductionExpr (EConst _) = pure []
    checkReductionExpr (EAdd lhs rhs) = (<>) <$> checkReductionExpr lhs <*> checkReductionExpr rhs
    checkReductionExpr (EMul lhs rhs) = (<>) <$> checkReductionExpr lhs <*> checkReductionExpr rhs
    checkReductionExpr (ELoad tensorName indices) = do
        ensureRankMatches tensorName indices
        let actual = ixExprName <$> indices
        forM_ actual $ \iter ->
            unless (Set.member iter knownIterNames) $
                Left $
                    ErrUnknownIndexIter iter
        pure [(tensorName, actual)]

firstDuplicate :: (Ord a) => [a] -> Maybe a
firstDuplicate = go Set.empty
  where
    go _ [] = Nothing
    go seen (x : xs)
        | Set.member x seen = Just x
        | otherwise = go (Set.insert x seen) xs

mkContext :: [ParamName] -> String
mkContext params
    | null params = "{ : }"
    | otherwise = withParams params $ "{ : " <> constraints <> " }"
  where
    constraints = intercalate " and " (mkNonNegative <$> params)

mkDomain :: [ParamName] -> [IterName] -> String
mkDomain params iters = withOptionalParams params body
  where
    body = "{ " <> statementRef iters <> " : " <> domainConstraints <> " }"
    domainConstraints = intercalate " and " (mkRange <$> zip iters params)

mkProgramOrder :: [ParamName] -> [IterName] -> String
mkProgramOrder params iters =
    withOptionalParams params $
        "{ " <> statementRef iters <> " -> " <> tuple iterNameToString iters <> " }"

mkReadAccess :: [ParamName] -> [IterName] -> [(TensorName, [IterName])] -> String
mkReadAccess _ _ [] = "{ }"
mkReadAccess params iters loads =
    withOptionalParams params $
        "{ " <> intercalate "; " relations <> " }"
  where
    relations =
        uniqueStable $
            (\(tensor, indices) -> statementRef iters <> " -> " <> tensorRef tensor indices) <$> loads

mkWriteAccess :: [ParamName] -> [IterName] -> (TensorName, [IterName]) -> String
mkWriteAccess params iters (tensor, indices) =
    withOptionalParams params $
        "{ "
            <> statementRef iters
            <> " -> "
            <> tensorRef tensor indices
            <> " }"

withOptionalParams :: [ParamName] -> String -> String
withOptionalParams [] body = body
withOptionalParams params body = withParams params body

withParams :: [ParamName] -> String -> String
withParams params body = tuple paramNameToString params <> " -> " <> body

tuple :: (a -> String) -> [a] -> String
tuple render xs = "[" <> intercalate "," (render <$> xs) <> "]"

statementRef :: [IterName] -> String
statementRef iters = "S" <> tuple iterNameToString iters

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

isOrderedSubsequence :: (Eq a) => [a] -> [a] -> Bool
isOrderedSubsequence [] _ = True
isOrderedSubsequence _ [] = False
isOrderedSubsequence (x : xs) (y : ys)
    | x == y = isOrderedSubsequence xs ys
    | otherwise = isOrderedSubsequence (x : xs) ys
