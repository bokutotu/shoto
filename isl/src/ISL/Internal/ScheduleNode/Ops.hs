module ISL.Internal.ScheduleNode.Ops (
    -- * Schedule Operations
    scheduleGetRoot,
    scheduleMapScheduleNodeBottomUp,

    -- * Schedule Node Operations
    scheduleNodeGetSchedule,
    scheduleNodeGetType,
    scheduleNodeNChildren,
    scheduleNodeChild,
    scheduleNodeParent,
    scheduleNodeRoot,
    scheduleNodeDelete,

    -- * Mark Operations
    scheduleNodeInsertMark,
    scheduleNodeMarkGetName,
    scheduleFindMarks,

    -- * Band Operations
    scheduleNodeBandNMember,
    scheduleNodeBandPermuteMembers,
    scheduleNodeBandTile,
) where

import           Control.Monad                   (unless, when, (>=>))
import           Control.Monad.IO.Class          (liftIO)
import           Foreign.C.String                (peekCString, withCString)
import           Foreign.ForeignPtr              (ForeignPtr, withForeignPtr)
import           Foreign.Ptr                     (nullPtr)
import           ISL.Core                        (Env (..), ISL, askEnv, manage,
                                                  throwISL)
import           ISL.Internal.FFI
import           ISL.Internal.Schedule.Types
import           ISL.Internal.ScheduleNode.Types

scheduleGetRoot :: Schedule s -> ISL s (ScheduleNode s)
scheduleGetRoot (Schedule schedFP) = do
    let mk = withForeignPtr schedFP c_sched_get_root
    manage c_sched_node_free "isl_schedule_get_root" mk ScheduleNode

scheduleNodeGetSchedule :: ScheduleNode s -> ISL s (Schedule s)
scheduleNodeGetSchedule (ScheduleNode nodeFP) = do
    let mk = withForeignPtr nodeFP c_sched_node_get_schedule
    manage c_sched_free "isl_schedule_node_get_schedule" mk Schedule

scheduleNodeGetType :: ScheduleNode s -> ISL s ScheduleNodeType
scheduleNodeGetType (ScheduleNode nodeFP) = do
    t <- liftIO $ withForeignPtr nodeFP c_sched_node_get_type
    if t == -1
        then throwISL "isl_schedule_node_get_type"
        else pure $ scheduleNodeTypeFromCInt t

scheduleNodeNChildren :: ScheduleNode s -> ISL s Int
scheduleNodeNChildren (ScheduleNode nodeFP) = do
    n <- liftIO $ withForeignPtr nodeFP c_sched_node_n_children
    if n < 0
        then throwISL "isl_schedule_node_n_children"
        else pure (fromIntegral n)

scheduleNodeChild :: ScheduleNode s -> Int -> ISL s (ScheduleNode s)
scheduleNodeChild (ScheduleNode nodeFP) pos = do
    when (pos < 0) $ throwISL "isl_schedule_node_child"
    let mk = withForeignPtr nodeFP $ \nodePtr -> do
            nodeCopy <- c_sched_node_copy nodePtr
            c_sched_node_child nodeCopy (fromIntegral pos)
    manage c_sched_node_free "isl_schedule_node_child" mk ScheduleNode

scheduleNodeParent :: ScheduleNode s -> ISL s (ScheduleNode s)
scheduleNodeParent (ScheduleNode nodeFP) = do
    let mk = withForeignPtr nodeFP $ \nodePtr -> do
            nodeCopy <- c_sched_node_copy nodePtr
            c_sched_node_parent nodeCopy
    manage c_sched_node_free "isl_schedule_node_parent" mk ScheduleNode

scheduleNodeRoot :: ScheduleNode s -> ISL s (ScheduleNode s)
scheduleNodeRoot (ScheduleNode nodeFP) = do
    let mk = withForeignPtr nodeFP $ \nodePtr -> do
            nodeCopy <- c_sched_node_copy nodePtr
            c_sched_node_root nodeCopy
    manage c_sched_node_free "isl_schedule_node_root" mk ScheduleNode

scheduleNodeDelete :: ScheduleNode s -> ISL s (ScheduleNode s)
scheduleNodeDelete (ScheduleNode nodeFP) = do
    let mk = withForeignPtr nodeFP $ \nodePtr -> do
            nodeCopy <- c_sched_node_copy nodePtr
            c_sched_node_delete nodeCopy
    manage c_sched_node_free "isl_schedule_node_delete" mk ScheduleNode

scheduleNodeInsertMark :: String -> ScheduleNode s -> ISL s (ScheduleNode s)
scheduleNodeInsertMark name (ScheduleNode nodeFP) = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctxPtr ->
            withCString name $ \cName ->
                withForeignPtr nodeFP $ \nodePtr -> do
                    idPtr <- c_id_alloc ctxPtr cName nullPtr
                    if idPtr == nullPtr
                        then pure nullPtr
                        else do
                            nodeCopy <- c_sched_node_copy nodePtr
                            c_sched_node_insert_mark nodeCopy idPtr
    manage c_sched_node_free "isl_schedule_node_insert_mark" mk ScheduleNode

scheduleNodeMarkGetName :: ScheduleNode s -> ISL s String
scheduleNodeMarkGetName (ScheduleNode nodeFP) = do
    idPtr <- liftIO $ withForeignPtr nodeFP c_sched_node_mark_get_id
    if idPtr == nullPtr
        then throwISL "isl_schedule_node_mark_get_id"
        else do
            cstr <- liftIO $ c_id_get_name idPtr
            name <-
                if cstr == nullPtr
                    then pure ""
                    else liftIO $ peekCString cstr
            liftIO $ c_id_free idPtr
            pure name

scheduleFindMarks :: String -> Schedule s -> ISL s [ScheduleNode s]
scheduleFindMarks markName sched = do
    root <- scheduleGetRoot sched
    go root
  where
    go node = do
        this <- do
            ty <- scheduleNodeGetType node
            case ty of
                ScheduleNodeMark -> do
                    name <- scheduleNodeMarkGetName node
                    pure ([node | name == markName])
                _ -> pure []
        n <- scheduleNodeNChildren node
        children <- mapM (scheduleNodeChild node >=> go) [0 .. n - 1]
        pure $ this ++ concat children

scheduleMapScheduleNodeBottomUp ::
    Schedule s ->
    (ScheduleNode s -> ISL s (ScheduleNode s)) ->
    ISL s (Schedule s)
scheduleMapScheduleNodeBottomUp sched f = do
    root <- scheduleGetRoot sched
    root' <- go root
    scheduleNodeGetSchedule root'
  where
    go node = do
        n <- scheduleNodeNChildren node
        node' <- foldl' (\m i -> m >>= step i) (pure node) [0 .. n - 1]
        f node'

    step i node = do
        child <- scheduleNodeChild node i
        child' <- go child
        scheduleNodeParent child'

-- Internal helper types for band transforms.
newtype Space s = Space (ForeignPtr IslSpace)

newtype MultiUnionPwAff s = MultiUnionPwAff (ForeignPtr IslMultiUnionPwAff)

newtype UnionPwAff s = UnionPwAff (ForeignPtr IslUnionPwAff)

newtype Val s = Val (ForeignPtr IslVal)

newtype MultiVal s = MultiVal (ForeignPtr IslMultiVal)

spaceRange :: Space s -> ISL s (Space s)
spaceRange (Space spaceFP) = do
    let mk = withForeignPtr spaceFP $ \spacePtr -> do
            spCopy <- c_space_copy spacePtr
            c_space_range spCopy
    manage c_space_free "isl_space_range" mk Space

mupaCopy :: MultiUnionPwAff s -> ISL s (MultiUnionPwAff s)
mupaCopy (MultiUnionPwAff mupaFP) = do
    let mk = withForeignPtr mupaFP c_mupa_copy
    manage c_mupa_free "isl_multi_union_pw_aff_copy" mk MultiUnionPwAff

mupaGetAt :: MultiUnionPwAff s -> Int -> ISL s (UnionPwAff s)
mupaGetAt (MultiUnionPwAff mupaFP) pos = do
    when (pos < 0) $ throwISL "isl_multi_union_pw_aff_get_at"
    let mk = withForeignPtr mupaFP $ \mupaPtr -> c_mupa_get_at mupaPtr (fromIntegral pos)
    manage c_upa_free "isl_multi_union_pw_aff_get_at" mk UnionPwAff

mupaSetAt :: MultiUnionPwAff s -> Int -> UnionPwAff s -> ISL s (MultiUnionPwAff s)
mupaSetAt (MultiUnionPwAff mupaFP) pos (UnionPwAff upaFP) = do
    when (pos < 0) $ throwISL "isl_multi_union_pw_aff_set_at"
    let mk =
            withForeignPtr mupaFP $ \mupaPtr ->
                withForeignPtr upaFP $ \upaPtr -> do
                    mupaC <- c_mupa_copy mupaPtr
                    upaC <- c_upa_copy upaPtr
                    c_mupa_set_at mupaC (fromIntegral pos) upaC
    manage c_mupa_free "isl_multi_union_pw_aff_set_at" mk MultiUnionPwAff

valIntFromSi :: Int -> ISL s (Val s)
valIntFromSi n = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP $ \ctxPtr -> c_val_int_from_si ctxPtr (fromIntegral n)
    manage c_val_free "isl_val_int_from_si" mk Val

multiValZero :: Space s -> ISL s (MultiVal s)
multiValZero (Space spaceFP) = do
    let mk = withForeignPtr spaceFP $ \spacePtr -> do
            spCopy <- c_space_copy spacePtr
            c_mval_zero spCopy
    manage c_mval_free "isl_multi_val_zero" mk MultiVal

multiValSetAt :: MultiVal s -> Int -> Val s -> ISL s (MultiVal s)
multiValSetAt (MultiVal mvFP) pos (Val vFP) = do
    when (pos < 0) $ throwISL "isl_multi_val_set_at"
    let mk =
            withForeignPtr mvFP $ \mvPtr ->
                withForeignPtr vFP $ \vPtr -> do
                    mvCopy <- c_mval_copy mvPtr
                    vCopy <- c_val_copy vPtr
                    c_mval_set_at mvCopy (fromIntegral pos) vCopy
    manage c_mval_free "isl_multi_val_set_at" mk MultiVal

scheduleNodeBandGetSpace :: ScheduleNode s -> ISL s (Space s)
scheduleNodeBandGetSpace (ScheduleNode nodeFP) = do
    let mk = withForeignPtr nodeFP c_sched_node_band_get_space
    manage c_space_free "isl_schedule_node_band_get_space" mk Space

scheduleNodeBandGetPartialSchedule :: ScheduleNode s -> ISL s (MultiUnionPwAff s)
scheduleNodeBandGetPartialSchedule (ScheduleNode nodeFP) = do
    let mk = withForeignPtr nodeFP c_sched_node_band_get_partial_schedule
    manage c_mupa_free "isl_schedule_node_band_get_partial_schedule" mk MultiUnionPwAff

scheduleNodeInsertPartialSchedule ::
    ScheduleNode s ->
    MultiUnionPwAff s ->
    ISL s (ScheduleNode s)
scheduleNodeInsertPartialSchedule (ScheduleNode nodeFP) (MultiUnionPwAff partialFP) = do
    let mk =
            withForeignPtr nodeFP $ \nodePtr ->
                withForeignPtr partialFP $ \partialPtr -> do
                    nodeCopy <- c_sched_node_copy nodePtr
                    partialCopy <- c_mupa_copy partialPtr
                    c_sched_node_insert_partial_schedule nodeCopy partialCopy
    manage c_sched_node_free "isl_schedule_node_insert_partial_schedule" mk ScheduleNode

scheduleNodeBandNMember :: ScheduleNode s -> ISL s Int
scheduleNodeBandNMember (ScheduleNode nodeFP) = do
    n <- liftIO $ withForeignPtr nodeFP c_sched_node_band_n_member
    if n < 0
        then throwISL "isl_schedule_node_band_n_member"
        else pure (fromIntegral n)

scheduleNodeBandPermuteMembers :: [Int] -> ScheduleNode s -> ISL s (ScheduleNode s)
scheduleNodeBandPermuteMembers perm node = do
    ty <- scheduleNodeGetType node
    unless (ty == ScheduleNodeBand) $ throwISL "scheduleNodeBandPermuteMembers"

    n <- scheduleNodeBandNMember node
    validatePermutation n perm

    partial <- scheduleNodeBandGetPartialSchedule node
    partial0 <- mupaCopy partial
    partial' <-
        foldl'
            ( \m (i, sourceIdx) ->
                m >>= \cur -> do
                    upa <- mupaGetAt partial sourceIdx
                    mupaSetAt cur i upa
            )
            (pure partial0)
            (zip [0 ..] perm)

    child <- scheduleNodeDelete node
    scheduleNodeInsertPartialSchedule child partial'
  where
    validatePermutation n xs = do
        when (length xs /= n) $ throwISL "scheduleNodeBandPermuteMembers"
        let okRange = all (\i -> 0 <= i && i < n) xs
        unless okRange $ throwISL "scheduleNodeBandPermuteMembers"
        let sorted = foldl' insert [] xs
        unless (sorted == [0 .. n - 1]) $ throwISL "scheduleNodeBandPermuteMembers"

    insert [] y = [y]
    insert (z : zs) y
        | y <= z = y : z : zs
        | otherwise = z : insert zs y

scheduleNodeBandTile :: [Int] -> ScheduleNode s -> ISL s (ScheduleNode s)
scheduleNodeBandTile sizes node = do
    ty <- scheduleNodeGetType node
    unless (ty == ScheduleNodeBand) $ throwISL "scheduleNodeBandTile"

    n <- scheduleNodeBandNMember node
    when (length sizes /= n) $ throwISL "scheduleNodeBandTile"

    bandSpace <- scheduleNodeBandGetSpace node
    rangeSpace <- spaceRange bandSpace

    mv0 <- multiValZero rangeSpace
    mv <-
        foldl'
            ( \m (i, sz) ->
                m >>= \mvAcc -> do
                    v <- valIntFromSi sz
                    multiValSetAt mvAcc i v
            )
            (pure mv0)
            (zip [0 ..] sizes)

    let ScheduleNode nodeFP = node
        MultiVal mvFP = mv
        mk =
            withForeignPtr nodeFP $ \nodePtr ->
                withForeignPtr mvFP $ \mvPtr -> do
                    nodeCopy <- c_sched_node_copy nodePtr
                    mvCopy <- c_mval_copy mvPtr
                    c_sched_node_band_tile nodeCopy mvCopy
    manage c_sched_node_free "isl_schedule_node_band_tile" mk ScheduleNode
