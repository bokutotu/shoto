{-# LANGUAGE MultiWayIf #-}

module ISL.Internal.Ast.Ops (
    -- * AST Build Operations
    astBuildAlloc,
    astBuildFromContext,
    astBuildNodeFromSchedule,

    -- * AST Node Operations
    astNodeToTree,
    astNodeToC,

    -- * AST Expression Operations
    astExprToExpression,
) where

import           Control.Exception           (bracket)
import           Control.Monad.IO.Class      (liftIO)
import           Foreign.C.String            (peekCString)
import           Foreign.C.Types             (CInt)
import qualified Foreign.Concurrent          as FC
import           Foreign.ForeignPtr          (ForeignPtr, touchForeignPtr,
                                              withForeignPtr)
import           Foreign.Marshal.Alloc       (free)
import           Foreign.Marshal.Utils       (toBool)
import           Foreign.Ptr                 (nullPtr)
import           ISL.Core                    (Env (..), ISL, askEnv, manage,
                                              throwISL)
import           ISL.Internal.Ast.Types      (AstBuild (..), AstExpr (..),
                                              AstExpression (..), AstNode (..),
                                              AstOp (..), AstTree (..))
import           ISL.Internal.FFI
import           ISL.Internal.Schedule.Types (Schedule (..))
import           ISL.Internal.Set.Types      (Set (..))

-- | Allocate a new AST build context
astBuildAlloc :: ISL s (AstBuild s)
astBuildAlloc = do
    Env ctxFP <- askEnv
    let mk = withForeignPtr ctxFP c_ast_build_alloc
    manage c_ast_build_free "isl_ast_build_alloc" mk AstBuild

-- | Create AST build from a context set
astBuildFromContext :: Set s -> ISL s (AstBuild s)
astBuildFromContext (Set setFP) = do
    let mk = withForeignPtr setFP $ \setPtr -> do
            setCopy <- c_set_copy setPtr
            c_ast_build_from_context setCopy
    manage c_ast_build_free "isl_ast_build_from_context" mk AstBuild

-- | Generate AST from a schedule
astBuildNodeFromSchedule :: AstBuild s -> Schedule s -> ISL s (AstNode s)
astBuildNodeFromSchedule (AstBuild buildFP) (Schedule schedFP) = do
    let mk = withForeignPtr buildFP $ \buildPtr ->
            withForeignPtr schedFP $ \schedPtr -> do
                schedCopy <- c_sched_copy schedPtr
                c_ast_build_node_from_schedule buildPtr schedCopy
    manage c_ast_node_free "isl_ast_build_node_from_schedule" mk AstNode

-- | Convert AST node to pure Haskell AstTree
astNodeToTree :: AstNode s -> ISL s AstTree
astNodeToTree (AstNode nodeFP) = walkNode nodeFP

-- | Convert AST node to C code string
astNodeToC :: AstNode s -> ISL s String
astNodeToC (AstNode nodeFP) = do
    Env ctxFP <- askEnv
    liftIO $ withForeignPtr ctxFP $ \ctx ->
        withForeignPtr nodeFP $ \nodePtr -> do
            printer <- c_printer_to_str ctx
            if printer == nullPtr
                then pure ""
                else do
                    printer' <- c_printer_set_output_format printer formatC
                    printer'' <- c_printer_print_ast_node printer' nodePtr
                    cstr <- c_printer_get_str printer''
                    str <-
                        if cstr == nullPtr
                            then pure ""
                            else bracket (pure cstr) free peekCString
                    c_printer_free printer''
                    pure str

-- | Convert AST expression wrapper to pure Haskell AstExpression
astExprToExpression :: AstExpr s -> ISL s AstExpression
astExprToExpression (AstExpr exprFP) = walkExpr exprFP

-- Internal helpers

walkNode :: ForeignPtr IslAstNode -> ISL s AstTree
walkNode nodeFP = do
    t <- liftIO $ withForeignPtr nodeFP c_ast_node_get_type
    if
        | t == astNodeTypeFor   -> walkForNode nodeFP
        | t == astNodeTypeIf    -> walkIfNode nodeFP
        | t == astNodeTypeBlock -> walkBlockNode nodeFP
        | t == astNodeTypeUser  -> walkUserNode nodeFP
        | t == astNodeTypeMark  -> walkMarkNode nodeFP
        | otherwise             -> pure AstError

walkForNode :: ForeignPtr IslAstNode -> ISL s AstTree
walkForNode nodeFP = do
    iterExpr <- getForIterator nodeFP
    initExpr <- getForInit nodeFP
    condExpr <- getForCond nodeFP
    incExpr <- getForInc nodeFP
    bodyNode <- getForBody nodeFP
    body <- walkNode bodyNode

    iterName <- case iterExpr of
        ExprId name -> pure name
        _           -> pure "<unknown>"

    pure $
        AstFor
            { forIterator = iterName
            , forInit = initExpr
            , forCond = condExpr
            , forInc = incExpr
            , forBody = body
            }
  where
    getForIterator fp = do
        exprFP <- manageExpr $ withForeignPtr fp c_ast_node_for_get_iterator
        walkExpr exprFP

    getForInit fp = do
        exprFP <- manageExpr $ withForeignPtr fp c_ast_node_for_get_init
        walkExpr exprFP

    getForCond fp = do
        exprFP <- manageExpr $ withForeignPtr fp c_ast_node_for_get_cond
        walkExpr exprFP

    getForInc fp = do
        exprFP <- manageExpr $ withForeignPtr fp c_ast_node_for_get_inc
        walkExpr exprFP

    getForBody fp =
        manageNode $ withForeignPtr fp c_ast_node_for_get_body

walkIfNode :: ForeignPtr IslAstNode -> ISL s AstTree
walkIfNode nodeFP = do
    condExpr <- getIfCond nodeFP
    thenNode <- getIfThen nodeFP
    thenTree <- walkNode thenNode

    hasElse <- liftIO $ withForeignPtr nodeFP c_ast_node_if_has_else_node
    elseTree <-
        if toBool hasElse
            then do
                elseNode <- getIfElse nodeFP
                Just <$> walkNode elseNode
            else pure Nothing

    pure $
        AstIf
            { ifCond = condExpr
            , ifThen = thenTree
            , ifElse = elseTree
            }
  where
    getIfCond fp = do
        exprFP <- manageExpr $ withForeignPtr fp c_ast_node_if_get_cond
        walkExpr exprFP

    getIfThen fp =
        manageNode $ withForeignPtr fp c_ast_node_if_get_then_node

    getIfElse fp =
        manageNode $ withForeignPtr fp c_ast_node_if_get_else_node

walkBlockNode :: ForeignPtr IslAstNode -> ISL s AstTree
walkBlockNode nodeFP = do
    listFP <- manageNodeList $ withForeignPtr nodeFP c_ast_node_block_get_children
    n <- liftIO $ withForeignPtr listFP c_ast_node_list_n_ast_node
    children <- mapM (getChildAt listFP) [0 .. n - 1]
    trees <- mapM walkNode children
    pure $ AstBlock trees
  where
    getChildAt listFP idx =
        manageNode $ withForeignPtr listFP $ \listPtr ->
            c_ast_node_list_get_at listPtr idx

walkUserNode :: ForeignPtr IslAstNode -> ISL s AstTree
walkUserNode nodeFP = do
    exprFP <- manageExpr $ withForeignPtr nodeFP c_ast_node_user_get_expr
    expr <- walkExpr exprFP
    pure $ AstUser expr

walkMarkNode :: ForeignPtr IslAstNode -> ISL s AstTree
walkMarkNode nodeFP = do
    markId <- getMarkId nodeFP
    childFP <- manageNode $ withForeignPtr nodeFP c_ast_node_mark_get_node
    child <- walkNode childFP
    pure $ AstMark markId child
  where
    getMarkId fp = liftIO $ withForeignPtr fp $ \p -> do
        idPtr <- c_ast_node_mark_get_id p
        if idPtr == nullPtr
            then pure ""
            else do
                cstr <- c_id_get_name idPtr
                name <-
                    if cstr == nullPtr
                        then pure ""
                        else peekCString cstr
                c_id_free idPtr
                pure name

walkExpr :: ForeignPtr IslAstExpr -> ISL s AstExpression
walkExpr exprFP = do
    t <- liftIO $ withForeignPtr exprFP c_ast_expr_get_type
    if
        | t == astExprTypeId  -> walkIdExpr exprFP
        | t == astExprTypeInt -> walkIntExpr exprFP
        | t == astExprTypeOp  -> walkOpExpr exprFP
        | otherwise           -> pure ExprError

walkIdExpr :: ForeignPtr IslAstExpr -> ISL s AstExpression
walkIdExpr exprFP = liftIO $ withForeignPtr exprFP $ \p -> do
    idPtr <- c_ast_expr_get_id p
    if idPtr == nullPtr
        then pure $ ExprId ""
        else do
            cstr <- c_id_get_name idPtr
            name <-
                if cstr == nullPtr
                    then pure ""
                    else peekCString cstr
            c_id_free idPtr
            pure $ ExprId name

walkIntExpr :: ForeignPtr IslAstExpr -> ISL s AstExpression
walkIntExpr exprFP = liftIO $ withForeignPtr exprFP $ \p -> do
    valPtr <- c_ast_expr_get_val p
    if valPtr == nullPtr
        then pure $ ExprInt 0
        else do
            n <- c_val_get_num_si valPtr
            c_val_free valPtr
            pure $ ExprInt (fromIntegral n)

walkOpExpr :: ForeignPtr IslAstExpr -> ISL s AstExpression
walkOpExpr exprFP = do
    opType <- liftIO $ withForeignPtr exprFP c_ast_expr_get_op_type
    nArgs <- liftIO $ withForeignPtr exprFP c_ast_expr_get_op_n_arg
    args <- mapM (getArg exprFP) [0 .. nArgs - 1]
    argExprs <- mapM walkExpr args
    pure $ ExprOp (cintToOp opType) argExprs
  where
    getArg fp idx =
        manageExpr $ withForeignPtr fp $ \p -> c_ast_expr_get_op_arg p idx

cintToOp :: CInt -> AstOp
cintToOp t
    | t == astOpAnd = OpAnd
    | t == astOpAndThen = OpAndThen
    | t == astOpOr = OpOr
    | t == astOpOrElse = OpOrElse
    | t == astOpMax = OpMax
    | t == astOpMin = OpMin
    | t == astOpMinus = OpMinus
    | t == astOpAdd = OpAdd
    | t == astOpSub = OpSub
    | t == astOpMul = OpMul
    | t == astOpDiv = OpDiv
    | t == astOpFdivQ = OpFdivQ
    | t == astOpPdivQ = OpPdivQ
    | t == astOpPdivR = OpPdivR
    | t == astOpZdivR = OpZdivR
    | t == astOpCond = OpCond
    | t == astOpSelect = OpSelect
    | t == astOpEq = OpEq
    | t == astOpLe = OpLe
    | t == astOpLt = OpLt
    | t == astOpGe = OpGe
    | t == astOpGt = OpGt
    | t == astOpCall = OpCall
    | t == astOpAccess = OpAccess
    | t == astOpMember = OpMember
    | t == astOpAddressOf = OpAddressOf
    | otherwise = OpUnknown (fromIntegral t)

-- Managed allocation helpers
manageNode :: IO RawAstNode -> ISL s (ForeignPtr IslAstNode)
manageNode mk = do
    Env ctxFP <- askEnv
    ptr <- liftIO mk
    if ptr == nullPtr
        then throwISL "ast_node"
        else liftIO $ do
            FC.newForeignPtr ptr $ do
                c_ast_node_free ptr
                touchForeignPtr ctxFP

manageExpr :: IO RawAstExpr -> ISL s (ForeignPtr IslAstExpr)
manageExpr mk = do
    Env ctxFP <- askEnv
    ptr <- liftIO mk
    if ptr == nullPtr
        then throwISL "ast_expr"
        else liftIO $ do
            FC.newForeignPtr ptr $ do
                c_ast_expr_free ptr
                touchForeignPtr ctxFP

manageNodeList :: IO RawAstNodeList -> ISL s (ForeignPtr IslAstNodeList)
manageNodeList mk = do
    Env ctxFP <- askEnv
    ptr <- liftIO mk
    if ptr == nullPtr
        then throwISL "ast_node_list"
        else liftIO $ do
            FC.newForeignPtr ptr $ do
                c_ast_node_list_free ptr
                touchForeignPtr ctxFP
