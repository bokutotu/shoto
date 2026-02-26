{-# LANGUAGE OverloadedRecordDot #-}

module FrontendIR.Lowering.Checker (
    checkProgram,
) where

import           Control.Monad               (forM_, unless)
import qualified Data.List.NonEmpty          as NE
import qualified Data.Map.Strict             as Map
import qualified Data.Set                    as Set
import           FrontendIR.Lowering.Checked (CheckedProgram (..),
                                              CheckedStmt (..))
import           FrontendIR.Types            (Axis (..), Expr (..),
                                              FrontendError (..), IterName,
                                              IxExpr (..), Program (..),
                                              Stmt (..), TensorDecl (..),
                                              TensorName)

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

ixExprName :: IxExpr -> IterName
ixExprName (IxVar name) = name

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
