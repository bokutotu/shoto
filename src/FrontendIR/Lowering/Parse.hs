{-# LANGUAGE OverloadedRecordDot #-}

module FrontendIR.Lowering.Parse (
    parseProgram,
) where

import           Control.Monad                    (foldM, forM_, unless, when,
                                                   zipWithM)
import           Data.List                        (isSubsequenceOf, (\\))
import qualified Data.List.NonEmpty               as NE
import qualified Data.Map.Strict                  as Map
import qualified Data.Set                         as Set
import           FrontendIR.Lowering.PolyhedralIR
import           FrontendIR.Types

data ParseEnv = ParseEnv
    { envParams :: [ParamName]
    , envIters :: [IterName]
    , envIterExtents :: Map.Map IterName ParamName
    , envKnownParams :: Set.Set ParamName
    , envKnownIters :: Set.Set IterName
    , envTensorRanks :: Map.Map TensorName Int
    }

parseProgram :: Program -> Either FrontendError PolyProgram
parseProgram prog = do
    env <- buildEnv prog

    let stmts = prog.stmts

    polyStmts <- case stmts of
        single NE.:| [] -> (: []) <$> parseSingleStmt env 0 single
        _ -> parseMultiStmtProgram env stmts
    pure
        PolyProgram
            { pParams = env.envParams
            , pIterExtents = env.envIterExtents
            , pStmts = polyStmts
            }

buildEnv :: Program -> Either FrontendError ParseEnv
buildEnv prog = do
    env1 <- foldM addAxis emptyEnv (NE.toList prog.axes)
    foldM addTensor env1 (NE.toList prog.tensors)
  where
    emptyEnv = ParseEnv [] [] Map.empty Set.empty Set.empty Map.empty

    addAxis env axis = do
        let i = axis.iter
            p = axis.extent
        when (i `Set.member` env.envKnownIters) $ Left $ ErrDuplicateIter i
        when (p `Set.member` env.envKnownParams) $ Left $ ErrDuplicateParam p
        pure
            env
                { envIters = env.envIters ++ [i]
                , envParams = env.envParams ++ [p]
                , envKnownIters = Set.insert i env.envKnownIters
                , envKnownParams = Set.insert p env.envKnownParams
                , envIterExtents = Map.insert i p env.envIterExtents
                }

    addTensor env decl = do
        let t = decl.tensor
        when (t `Map.member` env.envTensorRanks) $ Left $ ErrDuplicateTensor t
        forM_ decl.shape $ \p ->
            unless (p `Set.member` env.envKnownParams) $
                Left $
                    ErrUnknownTensorShapeParam t p
        pure
            env
                { envTensorRanks = Map.insert t (length decl.shape) env.envTensorRanks
                }

buildSchedAxes :: Int -> [IterName] -> [IterName] -> [ScheduleAxis]
buildSchedAxes stage domainIters allIters =
    let domainSet = Set.fromList domainIters
     in PadAxis stage : [if i `Set.member` domainSet then IterAxis i else PadAxis 0 | i <- allIters]

parseSingleStmt :: ParseEnv -> Int -> Stmt -> Either FrontendError PolyStmt
parseSingleStmt env sid stmt = do
    let actualStore = ixExprName <$> stmt.outputIndex
    ensureRankMatches env stmt.outputTensor actualStore

    let sName = StmtName ("S" <> show sid)
    let allIters = env.envIters

    case stmt of
        Assign{} -> do
            unless (actualStore == allIters) $
                Left $
                    ErrStoreIndexMismatch allIters actualStore

            let domainIters = actualStore
            let schedAxes = buildSchedAxes sid domainIters allIters
            readAccesses <- parseAssignExpr env sName domainIters allIters stmt.rhs
            let write = PolyAccess sName domainIters stmt.outputTensor actualStore

            pure
                PolyStmt
                    { stmtDomain = PolyDomain sName domainIters
                    , stmtSchedule = PolySchedule sName domainIters schedAxes
                    , stmtReads = readAccesses
                    , stmtWrite = write
                    , stmtIsReduction = False
                    }
        Reduction{} -> do
            ensureKnownIndexIters env actualStore
            unless (actualStore `isSubsequenceOf` allIters) $
                Left $
                    ErrReductionOutputNotSubsequence allIters actualStore

            let reducedAxes = allIters \\ actualStore
            when (null reducedAxes) $ Left ErrReductionRequiresReducedAxis

            let domainIters = actualStore ++ reducedAxes
            let schedAxes = buildSchedAxes sid domainIters allIters
            readAccesses <- parseReductionExpr env sName domainIters stmt.rhs
            let write = PolyAccess sName domainIters stmt.outputTensor actualStore

            pure
                PolyStmt
                    { stmtDomain = PolyDomain sName domainIters
                    , stmtSchedule = PolySchedule sName domainIters schedAxes
                    , stmtReads = readAccesses
                    , stmtWrite = write
                    , stmtIsReduction = True
                    }

parseMultiStmtProgram :: ParseEnv -> NE.NonEmpty Stmt -> Either FrontendError [PolyStmt]
parseMultiStmtProgram env stmts = do
    -- TODO: This is a known limitation. Multi-stmt lowering assumes all statements
    -- share the first statement's output indices; this validation should move to
    -- the early checking boundary so parsing can focus on lowering only.
    let stmtsList = NE.toList stmts
    spatialIters <- ensureSameOutputIndices stmts
    ensureKnownIndexIters env spatialIters
    unless (spatialIters `isSubsequenceOf` env.envIters) $
        Left $
            ErrStoreIndexMismatch env.envIters spatialIters

    zipWithM (parseMultiStmt env spatialIters) [0 ..] stmtsList

ensureSameOutputIndices :: NE.NonEmpty Stmt -> Either FrontendError [IterName]
ensureSameOutputIndices (firstStmt NE.:| restStmts) = do
    let expectedStore = ixExprName <$> firstStmt.outputIndex
    forM_ restStmts $ \stmt -> do
        let actualStore = ixExprName <$> stmt.outputIndex
        unless (actualStore == expectedStore) $
            Left $
                ErrStoreIndexMismatch expectedStore actualStore
    pure expectedStore

parseMultiStmt :: ParseEnv -> [IterName] -> Int -> Stmt -> Either FrontendError PolyStmt
parseMultiStmt env spatialIters sid stmt = do
    let actualStore = ixExprName <$> stmt.outputIndex

    ensureRankMatches env stmt.outputTensor actualStore

    let sName = StmtName ("S" <> show sid)
    let allIters = env.envIters

    case stmt of
        Assign{} -> do
            let domainIters = actualStore
            let schedAxes = buildSchedAxes sid domainIters allIters
            readAccesses <- parseAssignExpr env sName domainIters spatialIters stmt.rhs
            let write = PolyAccess sName domainIters stmt.outputTensor actualStore

            pure
                PolyStmt
                    { stmtDomain = PolyDomain sName domainIters
                    , stmtSchedule = PolySchedule sName domainIters schedAxes
                    , stmtReads = readAccesses
                    , stmtWrite = write
                    , stmtIsReduction = False
                    }
        Reduction{} -> do
            let reducedAxes = allIters \\ spatialIters
            when (null reducedAxes) $ Left ErrReductionRequiresReducedAxis
            unless (length reducedAxes == 1) $
                Left $
                    ErrMultiStmtRequiresSingleReductionAxis reducedAxes

            let domainIters = actualStore ++ reducedAxes
            let schedAxes = buildSchedAxes sid domainIters allIters
            readAccesses <- parseReductionExpr env sName domainIters stmt.rhs
            let write = PolyAccess sName domainIters stmt.outputTensor actualStore

            pure
                PolyStmt
                    { stmtDomain = PolyDomain sName domainIters
                    , stmtSchedule = PolySchedule sName domainIters schedAxes
                    , stmtReads = readAccesses
                    , stmtWrite = write
                    , stmtIsReduction = True
                    }

parseAssignExpr ::
    ParseEnv -> StmtName -> [IterName] -> [IterName] -> Expr -> Either FrontendError [PolyAccess]
parseAssignExpr env sName domainIters expectedIndices = go
  where
    go (EConst _) = pure []
    go (EAdd lhs rhs) = (<>) <$> go lhs <*> go rhs
    go (EMul lhs rhs) = (<>) <$> go lhs <*> go rhs
    go (ELoad tensor indices) = do
        let actual = ixExprName <$> indices
        ensureRankMatches env tensor actual
        unless (actual == expectedIndices) $
            Left $
                ErrLoadIndexMismatch tensor expectedIndices actual
        pure [PolyAccess sName domainIters tensor actual]

parseReductionExpr ::
    ParseEnv -> StmtName -> [IterName] -> Expr -> Either FrontendError [PolyAccess]
parseReductionExpr env sName domainIters = go
  where
    go (EConst _) = pure []
    go (EAdd lhs rhs) = (<>) <$> go lhs <*> go rhs
    go (EMul lhs rhs) = (<>) <$> go lhs <*> go rhs
    go (ELoad tensor indices) = do
        let actual = ixExprName <$> indices
        ensureRankMatches env tensor actual
        ensureKnownIndexIters env actual
        pure [PolyAccess sName domainIters tensor actual]

ensureRankMatches :: ParseEnv -> TensorName -> [IterName] -> Either FrontendError ()
ensureRankMatches env tensor indices =
    case Map.lookup tensor env.envTensorRanks of
        Nothing -> Left $ ErrUndeclaredTensor tensor
        Just expectedRank ->
            unless (length indices == expectedRank) $
                Left $
                    ErrTensorRankMismatch tensor expectedRank (length indices)

ensureKnownIndexIters :: ParseEnv -> [IterName] -> Either FrontendError ()
ensureKnownIndexIters env iters =
    forM_ iters $ \iter ->
        unless (iter `Set.member` env.envKnownIters) $
            Left $
                ErrUnknownIndexIter iter

ixExprName :: IxExpr -> IterName
ixExprName (IxVar name) = name
