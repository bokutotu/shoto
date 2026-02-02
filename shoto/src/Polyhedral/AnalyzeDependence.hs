{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE RecordWildCards     #-}

module Polyhedral.AnalyzeDependence (
    DependenceAnalysis (..),
    analyzeDependences,
    toScheduleConstraints,
) where

import           ISL              (ISL, unionAccessInfoComputeFlow,
                                   unionAccessInfoFromSink,
                                   unionAccessInfoSetMaySource,
                                   unionAccessInfoSetScheduleMap,
                                   unionFlowGetMayDependence, unionMapSubtract)
import           Polyhedral.Empty (emptyMap, isEmptyMap, isEmptySet)
import           Polyhedral.Types (Access (..), Dependencies (..),
                                   Dependency (..), Domain,
                                   FromUnionMap (fromUnionMap),
                                   IntoUnionMap (intoUnionMap),
                                   PolyhedralModel (..), ProgramOrder (..),
                                   ReadMap, WriteMap)
import           Polyhedral.Unite (uniteMap)

-- | 依存関係解析の生結果
data DependenceAnalysis s = DependenceAnalysis
    { raw              :: Dependency s
    -- ^ RAW依存
    , reductionCarried :: Dependency s
    -- ^ リダクション内部の依存
    , validityBase     :: Dependency s
    -- ^ 全依存 - リダクション依存
    }

mayDeps :: Access t1 s -> Access t2 s -> ProgramOrder s -> ISL s (Dependency s)
mayDeps source sink po = do
    anyEmpty <- or <$> sequence [isEmptyMap source, isEmptyMap sink, isEmptyMap po]
    if anyEmpty then emptyMap else may
  where
    may = do
        info <- unionAccessInfoFromSink (intoUnionMap sink)
        info' <- unionAccessInfoSetMaySource info (intoUnionMap source)
        info'' <- unionAccessInfoSetScheduleMap info' (intoUnionMap po)
        flow <- unionAccessInfoComputeFlow info''
        Dependency <$> unionFlowGetMayDependence flow

reductionCarriedDeps ::
    forall s.
    Domain s -> Access ReadMap s -> Access WriteMap s -> ProgramOrder s -> ISL s (Dependency s)
reductionCarriedDeps dom redRead redWrite po = do
    domWriteEmpty <- or <$> sequence [isEmptySet dom, isEmptyMap redWrite]
    if domWriteEmpty then emptyMap else deps
  where
    wawDeps = mayDeps redWrite redWrite po
    reductionDeps = do
        raw <- mayDeps redWrite redRead po
        waw <- wawDeps
        dep <- (raw `uniteMap` waw) :: ISL s (Dependency s)
        war <- mayDeps redRead redWrite po
        dep `uniteMap` war
    deps = do
        readEmpty <- isEmptyMap redRead
        if readEmpty then wawDeps else reductionDeps

analyzeDependences :: forall s. PolyhedralModel s -> ISL s (DependenceAnalysis s)
analyzeDependences model = do
    raw <- mayDeps model.writeAccess model.readAccess model.programOrder
    waw <- mayDeps model.writeAccess model.writeAccess model.programOrder
    war <- mayDeps model.readAccess model.writeAccess model.programOrder
    allDeps <- (uniteMap raw waw :: ISL s (Dependency s)) >>= uniteMap war :: ISL s (Dependency s)
    reductionCarried <-
        reductionCarriedDeps
            model.reductionDomain
            model.reductionRead
            model.reductionWrite
            model.programOrder
    validityBase <-
        fromUnionMap
            <$> unionMapSubtract (intoUnionMap allDeps) (intoUnionMap reductionCarried)
    pure DependenceAnalysis{..}

-- | 解析結果からスケジューリング制約に変換
toScheduleConstraints :: DependenceAnalysis s -> ISL s (Dependencies s)
toScheduleConstraints analysis = do
    prox <- uniteMap analysis.raw analysis.reductionCarried
    pure
        Dependencies
            { validity = analysis.validityBase
            , coincidence = analysis.validityBase
            , proximity = prox
            }
