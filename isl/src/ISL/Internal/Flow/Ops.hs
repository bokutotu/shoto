module ISL.Internal.Flow.Ops (
    -- * UnionAccessInfo Operations
    unionAccessInfoFromSink,
    unionAccessInfoSetMustSource,
    unionAccessInfoSetMaySource,
    unionAccessInfoSetScheduleMap,
    unionAccessInfoComputeFlow,

    -- * UnionFlow Operations
    unionFlowGetMustDependence,
    unionFlowGetMayDependence,
) where

import           Foreign.ForeignPtr      (withForeignPtr)
import           ISL.Core                (ISL, manage)
import           ISL.Internal.FFI
import           ISL.Internal.Flow.Types (UnionAccessInfo (..), UnionFlow (..))
import           ISL.Internal.Map.Types  (UnionMap (..))

-- | Create access info from sink (reads)
unionAccessInfoFromSink :: UnionMap s -> ISL s (UnionAccessInfo s)
unionAccessInfoFromSink (UnionMap fp) = do
    let mk = withForeignPtr fp $ \ptr -> do
            cptr <- c_umap_copy ptr
            c_union_access_info_from_sink cptr
    manage c_union_access_info_free "isl_union_access_info_from_sink" mk UnionAccessInfo

-- | Set must source (writes for RAW dependence)
unionAccessInfoSetMustSource :: UnionAccessInfo s -> UnionMap s -> ISL s (UnionAccessInfo s)
unionAccessInfoSetMustSource (UnionAccessInfo aiFP) (UnionMap srcFP) = do
    let mk = withForeignPtr aiFP $ \aiPtr ->
            withForeignPtr srcFP $ \srcPtr -> do
                aiCopy <- c_union_access_info_copy aiPtr
                srcCopy <- c_umap_copy srcPtr
                c_union_access_info_set_must_source aiCopy srcCopy
    manage c_union_access_info_free "isl_union_access_info_set_must_source" mk UnionAccessInfo

-- | Set may source
unionAccessInfoSetMaySource :: UnionAccessInfo s -> UnionMap s -> ISL s (UnionAccessInfo s)
unionAccessInfoSetMaySource (UnionAccessInfo aiFP) (UnionMap srcFP) = do
    let mk = withForeignPtr aiFP $ \aiPtr ->
            withForeignPtr srcFP $ \srcPtr -> do
                aiCopy <- c_union_access_info_copy aiPtr
                srcCopy <- c_umap_copy srcPtr
                c_union_access_info_set_may_source aiCopy srcCopy
    manage c_union_access_info_free "isl_union_access_info_set_may_source" mk UnionAccessInfo

-- | Set schedule map
unionAccessInfoSetScheduleMap :: UnionAccessInfo s -> UnionMap s -> ISL s (UnionAccessInfo s)
unionAccessInfoSetScheduleMap (UnionAccessInfo aiFP) (UnionMap schedFP) = do
    let mk = withForeignPtr aiFP $ \aiPtr ->
            withForeignPtr schedFP $ \schedPtr -> do
                aiCopy <- c_union_access_info_copy aiPtr
                schedCopy <- c_umap_copy schedPtr
                c_union_access_info_set_schedule_map aiCopy schedCopy
    manage c_union_access_info_free "isl_union_access_info_set_schedule_map" mk UnionAccessInfo

-- | Compute flow (dependence analysis)
unionAccessInfoComputeFlow :: UnionAccessInfo s -> ISL s (UnionFlow s)
unionAccessInfoComputeFlow (UnionAccessInfo aiFP) = do
    let mk = withForeignPtr aiFP $ \aiPtr -> do
            aiCopy <- c_union_access_info_copy aiPtr
            c_union_access_info_compute_flow aiCopy
    manage c_union_flow_free "isl_union_access_info_compute_flow" mk UnionFlow

-- | Get must dependence from flow
unionFlowGetMustDependence :: UnionFlow s -> ISL s (UnionMap s)
unionFlowGetMustDependence (UnionFlow fp) = do
    let mk = withForeignPtr fp c_union_flow_get_must_dependence
    manage c_umap_free "isl_union_flow_get_must_dependence" mk UnionMap

-- | Get may dependence from flow
unionFlowGetMayDependence :: UnionFlow s -> ISL s (UnionMap s)
unionFlowGetMayDependence (UnionFlow fp) = do
    let mk = withForeignPtr fp c_union_flow_get_may_dependence
    manage c_umap_free "isl_union_flow_get_may_dependence" mk UnionMap
