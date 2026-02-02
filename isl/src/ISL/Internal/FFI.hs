{-# LANGUAGE ForeignFunctionInterface #-}

module ISL.Internal.FFI (
    -- * Raw Pointer Types
    IslCtx,
    IslSet,
    IslUnionSet,
    IslMap,
    IslUnionMap,
    IslSchedule,
    IslScheduleConstraints,
    IslId,
    IslAstBuild,
    IslAstNode,
    IslAstExpr,
    IslAstNodeList,
    IslVal,
    IslPrinter,
    IslUnionAccessInfo,
    IslUnionFlow,
    RawCtx,
    RawSet,
    RawUnionSet,
    RawMap,
    RawUnionMap,
    RawSchedule,
    RawScheduleConstraints,
    RawId,
    RawAstBuild,
    RawAstNode,
    RawAstExpr,
    RawAstNodeList,
    RawVal,
    RawPrinter,
    RawUnionAccessInfo,
    RawUnionFlow,

    -- * Context FFI
    c_ctx_alloc,
    p_ctx_free,
    c_ctx_last_error_msg,
    c_ctx_last_error_file,
    c_ctx_last_error_line,

    -- * Set FFI
    c_set_read,
    c_set_to_str,
    c_set_free,
    c_set_copy,
    c_set_union,
    c_set_intersect,
    c_set_subtract,
    c_set_coalesce,
    c_set_is_equal,

    -- * Union Set FFI
    c_uset_read,
    c_uset_to_str,
    c_uset_free,
    c_uset_copy,
    c_uset_union,
    c_uset_intersect,
    c_uset_subtract,
    c_uset_coalesce,
    c_uset_is_equal,
    c_uset_is_empty,

    -- * Map FFI
    c_map_read,
    c_map_to_str,
    c_map_free,
    c_map_copy,
    c_map_union,
    c_map_intersect,
    c_map_subtract,
    c_map_coalesce,
    c_map_is_equal,
    c_map_domain,
    c_map_range,
    c_map_reverse,
    c_map_apply_range,
    c_map_apply_domain,

    -- * Union Map FFI
    c_umap_read,
    c_umap_to_str,
    c_umap_free,
    c_umap_copy,
    c_umap_union,
    c_umap_intersect,
    c_umap_subtract,
    c_umap_coalesce,
    c_umap_is_equal,
    c_umap_is_empty,
    c_umap_domain,
    c_umap_range,
    c_umap_reverse,
    c_umap_apply_range,
    c_umap_apply_domain,
    c_umap_intersect_domain,
    c_umap_lex_lt_union_map,

    -- * Schedule FFI
    c_sched_read,
    c_sched_to_str,
    c_sched_free,
    c_sched_copy,
    c_sched_from_domain,
    c_sched_get_domain,
    c_sched_plain_is_equal,

    -- * Schedule Constraints FFI
    c_sched_constraints_on_domain,
    c_sched_constraints_set_validity,
    c_sched_constraints_set_proximity,
    c_sched_constraints_set_coincidence,
    c_sched_constraints_compute_schedule,
    c_sched_constraints_free,

    -- * Union Access Info FFI
    c_union_access_info_from_sink,
    c_union_access_info_copy,
    c_union_access_info_set_must_source,
    c_union_access_info_set_may_source,
    c_union_access_info_set_schedule_map,
    c_union_access_info_compute_flow,
    c_union_access_info_free,

    -- * Union Flow FFI
    c_union_flow_get_must_dependence,
    c_union_flow_get_may_dependence,
    c_union_flow_free,

    -- * ID FFI
    c_id_free,
    c_id_get_name,

    -- * AST Build FFI
    c_ast_build_alloc,
    c_ast_build_from_context,
    c_ast_build_node_from_schedule,
    c_ast_build_free,

    -- * AST Node FFI
    c_ast_node_get_type,
    c_ast_node_free,
    c_ast_node_copy,
    c_ast_node_for_get_iterator,
    c_ast_node_for_get_init,
    c_ast_node_for_get_cond,
    c_ast_node_for_get_inc,
    c_ast_node_for_get_body,
    c_ast_node_if_get_cond,
    c_ast_node_if_get_then_node,
    c_ast_node_if_has_else_node,
    c_ast_node_if_get_else_node,
    c_ast_node_block_get_children,
    c_ast_node_user_get_expr,
    c_ast_node_mark_get_id,
    c_ast_node_mark_get_node,

    -- * AST Node List FFI
    c_ast_node_list_n_ast_node,
    c_ast_node_list_get_at,
    c_ast_node_list_free,

    -- * AST Expr FFI
    c_ast_expr_get_type,
    c_ast_expr_free,
    c_ast_expr_copy,
    c_ast_expr_get_id,
    c_ast_expr_get_val,
    c_ast_expr_get_op_type,
    c_ast_expr_get_op_n_arg,
    c_ast_expr_get_op_arg,

    -- * Val FFI
    c_val_get_num_si,
    c_val_free,

    -- * Printer FFI
    c_printer_to_str,
    c_printer_set_output_format,
    c_printer_print_ast_node,
    c_printer_get_str,
    c_printer_free,

    -- * AST Node Type Constants
    astNodeTypeError,
    astNodeTypeFor,
    astNodeTypeIf,
    astNodeTypeBlock,
    astNodeTypeMark,
    astNodeTypeUser,

    -- * AST Expr Type Constants
    astExprTypeError,
    astExprTypeId,
    astExprTypeInt,
    astExprTypeOp,

    -- * AST Op Type Constants
    astOpError,
    astOpAnd,
    astOpAndThen,
    astOpOr,
    astOpOrElse,
    astOpMax,
    astOpMin,
    astOpMinus,
    astOpAdd,
    astOpSub,
    astOpMul,
    astOpDiv,
    astOpFdivQ,
    astOpPdivQ,
    astOpPdivR,
    astOpZdivR,
    astOpCond,
    astOpSelect,
    astOpEq,
    astOpLe,
    astOpLt,
    astOpGe,
    astOpGt,
    astOpCall,
    astOpAccess,
    astOpMember,
    astOpAddressOf,

    -- * Printer Format Constants
    formatIsl,
    formatC,
) where

import           Foreign.C.String (CString)
import           Foreign.C.Types  (CInt (..), CLong (..))
import           Foreign.Ptr      (FunPtr, Ptr)

-- Raw pointer types
data IslCtx

data IslSet

data IslUnionSet

data IslMap

data IslUnionMap

data IslSchedule

data IslScheduleConstraints

data IslId

data IslAstBuild

data IslAstNode

data IslAstExpr

data IslAstNodeList

data IslVal

data IslPrinter

data IslUnionAccessInfo

data IslUnionFlow

type RawCtx = Ptr IslCtx

type RawSet = Ptr IslSet

type RawUnionSet = Ptr IslUnionSet

type RawMap = Ptr IslMap

type RawUnionMap = Ptr IslUnionMap

type RawSchedule = Ptr IslSchedule

type RawScheduleConstraints = Ptr IslScheduleConstraints

type RawId = Ptr IslId

type RawAstBuild = Ptr IslAstBuild

type RawAstNode = Ptr IslAstNode

type RawAstExpr = Ptr IslAstExpr

type RawAstNodeList = Ptr IslAstNodeList

type RawVal = Ptr IslVal

type RawPrinter = Ptr IslPrinter

type RawUnionAccessInfo = Ptr IslUnionAccessInfo

type RawUnionFlow = Ptr IslUnionFlow

-- Context
foreign import ccall "isl/ctx.h isl_ctx_alloc"
    c_ctx_alloc :: IO RawCtx

foreign import ccall "isl/ctx.h &isl_ctx_free"
    p_ctx_free :: FunPtr (RawCtx -> IO ())

foreign import ccall "isl/ctx.h isl_ctx_last_error_msg"
    c_ctx_last_error_msg :: RawCtx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_file"
    c_ctx_last_error_file :: RawCtx -> IO CString

foreign import ccall "isl/ctx.h isl_ctx_last_error_line"
    c_ctx_last_error_line :: RawCtx -> IO CInt

-- Set
foreign import ccall "isl/set.h isl_set_read_from_str"
    c_set_read :: RawCtx -> CString -> IO RawSet

foreign import ccall "isl/set.h isl_set_to_str"
    c_set_to_str :: RawSet -> IO CString

foreign import ccall "isl/set.h isl_set_free"
    c_set_free :: RawSet -> IO ()

foreign import ccall "isl/set.h isl_set_copy"
    c_set_copy :: RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_union"
    c_set_union :: RawSet -> RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_intersect"
    c_set_intersect :: RawSet -> RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_subtract"
    c_set_subtract :: RawSet -> RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_coalesce"
    c_set_coalesce :: RawSet -> IO RawSet

foreign import ccall "isl/set.h isl_set_is_equal"
    c_set_is_equal :: RawSet -> RawSet -> IO CInt

-- Union Set
foreign import ccall "isl/union_set.h isl_union_set_read_from_str"
    c_uset_read :: RawCtx -> CString -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_to_str"
    c_uset_to_str :: RawUnionSet -> IO CString

foreign import ccall "isl/union_set.h isl_union_set_free"
    c_uset_free :: RawUnionSet -> IO ()

foreign import ccall "isl/union_set.h isl_union_set_copy"
    c_uset_copy :: RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_union"
    c_uset_union :: RawUnionSet -> RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_intersect"
    c_uset_intersect :: RawUnionSet -> RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_subtract"
    c_uset_subtract :: RawUnionSet -> RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_coalesce"
    c_uset_coalesce :: RawUnionSet -> IO RawUnionSet

foreign import ccall "isl/union_set.h isl_union_set_is_equal"
    c_uset_is_equal :: RawUnionSet -> RawUnionSet -> IO CInt

foreign import ccall "isl/union_set.h isl_union_set_is_empty"
    c_uset_is_empty :: RawUnionSet -> IO CInt

-- Map
foreign import ccall "isl/map.h isl_map_read_from_str"
    c_map_read :: RawCtx -> CString -> IO RawMap

foreign import ccall "isl/map.h isl_map_to_str"
    c_map_to_str :: RawMap -> IO CString

foreign import ccall "isl/map.h isl_map_free"
    c_map_free :: RawMap -> IO ()

foreign import ccall "isl/map.h isl_map_copy"
    c_map_copy :: RawMap -> IO RawMap

foreign import ccall "isl/map.h isl_map_union"
    c_map_union :: RawMap -> RawMap -> IO RawMap

foreign import ccall "isl/map.h isl_map_intersect"
    c_map_intersect :: RawMap -> RawMap -> IO RawMap

foreign import ccall "isl/map.h isl_map_subtract"
    c_map_subtract :: RawMap -> RawMap -> IO RawMap

foreign import ccall "isl/map.h isl_map_coalesce"
    c_map_coalesce :: RawMap -> IO RawMap

foreign import ccall "isl/map.h isl_map_is_equal"
    c_map_is_equal :: RawMap -> RawMap -> IO CInt

foreign import ccall "isl/map.h isl_map_domain"
    c_map_domain :: RawMap -> IO RawSet

foreign import ccall "isl/map.h isl_map_range"
    c_map_range :: RawMap -> IO RawSet

foreign import ccall "isl/map.h isl_map_reverse"
    c_map_reverse :: RawMap -> IO RawMap

foreign import ccall "isl/map.h isl_map_apply_range"
    c_map_apply_range :: RawMap -> RawMap -> IO RawMap

foreign import ccall "isl/map.h isl_map_apply_domain"
    c_map_apply_domain :: RawMap -> RawMap -> IO RawMap

-- Union Map
foreign import ccall "isl/union_map.h isl_union_map_read_from_str"
    c_umap_read :: RawCtx -> CString -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_to_str"
    c_umap_to_str :: RawUnionMap -> IO CString

foreign import ccall "isl/union_map.h isl_union_map_free"
    c_umap_free :: RawUnionMap -> IO ()

foreign import ccall "isl/union_map.h isl_union_map_copy"
    c_umap_copy :: RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_union"
    c_umap_union :: RawUnionMap -> RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_intersect"
    c_umap_intersect :: RawUnionMap -> RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_subtract"
    c_umap_subtract :: RawUnionMap -> RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_coalesce"
    c_umap_coalesce :: RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_is_equal"
    c_umap_is_equal :: RawUnionMap -> RawUnionMap -> IO CInt

foreign import ccall "isl/union_map.h isl_union_map_is_empty"
    c_umap_is_empty :: RawUnionMap -> IO CInt

foreign import ccall "isl/union_map.h isl_union_map_domain"
    c_umap_domain :: RawUnionMap -> IO RawUnionSet

foreign import ccall "isl/union_map.h isl_union_map_range"
    c_umap_range :: RawUnionMap -> IO RawUnionSet

foreign import ccall "isl/union_map.h isl_union_map_reverse"
    c_umap_reverse :: RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_apply_range"
    c_umap_apply_range :: RawUnionMap -> RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_apply_domain"
    c_umap_apply_domain :: RawUnionMap -> RawUnionMap -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_intersect_domain"
    c_umap_intersect_domain :: RawUnionMap -> RawUnionSet -> IO RawUnionMap

foreign import ccall "isl/union_map.h isl_union_map_lex_lt_union_map"
    c_umap_lex_lt_union_map :: RawUnionMap -> RawUnionMap -> IO RawUnionMap

-- Schedule
foreign import ccall "isl/schedule.h isl_schedule_read_from_str"
    c_sched_read :: RawCtx -> CString -> IO RawSchedule

foreign import ccall "isl/schedule.h isl_schedule_to_str"
    c_sched_to_str :: RawSchedule -> IO CString

foreign import ccall "isl/schedule.h isl_schedule_free"
    c_sched_free :: RawSchedule -> IO ()

foreign import ccall "isl/schedule.h isl_schedule_copy"
    c_sched_copy :: RawSchedule -> IO RawSchedule

foreign import ccall "isl/schedule.h isl_schedule_from_domain"
    c_sched_from_domain :: RawUnionSet -> IO RawSchedule

foreign import ccall "isl/schedule.h isl_schedule_get_domain"
    c_sched_get_domain :: RawSchedule -> IO RawUnionSet

foreign import ccall "isl/schedule.h isl_schedule_plain_is_equal"
    c_sched_plain_is_equal :: RawSchedule -> RawSchedule -> IO CInt

-- Schedule Constraints Operations
foreign import ccall "isl/schedule.h isl_schedule_constraints_on_domain"
    c_sched_constraints_on_domain :: RawUnionSet -> IO RawScheduleConstraints

foreign import ccall "isl/schedule.h isl_schedule_constraints_set_validity"
    c_sched_constraints_set_validity ::
        RawScheduleConstraints -> RawUnionMap -> IO RawScheduleConstraints

foreign import ccall "isl/schedule.h isl_schedule_constraints_set_proximity"
    c_sched_constraints_set_proximity ::
        RawScheduleConstraints -> RawUnionMap -> IO RawScheduleConstraints

foreign import ccall "isl/schedule.h isl_schedule_constraints_set_coincidence"
    c_sched_constraints_set_coincidence ::
        RawScheduleConstraints -> RawUnionMap -> IO RawScheduleConstraints

foreign import ccall "isl/schedule.h isl_schedule_constraints_compute_schedule"
    c_sched_constraints_compute_schedule :: RawScheduleConstraints -> IO RawSchedule

foreign import ccall "isl/schedule.h isl_schedule_constraints_free"
    c_sched_constraints_free :: RawScheduleConstraints -> IO ()

-- Union Access Info Operations
foreign import ccall "isl/flow.h isl_union_access_info_from_sink"
    c_union_access_info_from_sink :: RawUnionMap -> IO RawUnionAccessInfo

foreign import ccall "isl/flow.h isl_union_access_info_copy"
    c_union_access_info_copy :: RawUnionAccessInfo -> IO RawUnionAccessInfo

foreign import ccall "isl/flow.h isl_union_access_info_set_must_source"
    c_union_access_info_set_must_source ::
        RawUnionAccessInfo -> RawUnionMap -> IO RawUnionAccessInfo

foreign import ccall "isl/flow.h isl_union_access_info_set_may_source"
    c_union_access_info_set_may_source ::
        RawUnionAccessInfo -> RawUnionMap -> IO RawUnionAccessInfo

foreign import ccall "isl/flow.h isl_union_access_info_set_schedule_map"
    c_union_access_info_set_schedule_map ::
        RawUnionAccessInfo -> RawUnionMap -> IO RawUnionAccessInfo

foreign import ccall "isl/flow.h isl_union_access_info_compute_flow"
    c_union_access_info_compute_flow :: RawUnionAccessInfo -> IO RawUnionFlow

foreign import ccall "isl/flow.h isl_union_access_info_free"
    c_union_access_info_free :: RawUnionAccessInfo -> IO ()

-- Union Flow Operations
foreign import ccall "isl/flow.h isl_union_flow_get_must_dependence"
    c_union_flow_get_must_dependence :: RawUnionFlow -> IO RawUnionMap

foreign import ccall "isl/flow.h isl_union_flow_get_may_dependence"
    c_union_flow_get_may_dependence :: RawUnionFlow -> IO RawUnionMap

foreign import ccall "isl/flow.h isl_union_flow_free"
    c_union_flow_free :: RawUnionFlow -> IO ()

-- ID Operations
foreign import ccall "isl/id.h isl_id_free"
    c_id_free :: RawId -> IO ()

foreign import ccall "isl/id.h isl_id_get_name"
    c_id_get_name :: RawId -> IO CString

-- AST Build Operations
foreign import ccall "isl/ast_build.h isl_ast_build_alloc"
    c_ast_build_alloc :: RawCtx -> IO RawAstBuild

foreign import ccall "isl/ast_build.h isl_ast_build_from_context"
    c_ast_build_from_context :: RawSet -> IO RawAstBuild

foreign import ccall "isl/ast_build.h isl_ast_build_node_from_schedule"
    c_ast_build_node_from_schedule :: RawAstBuild -> RawSchedule -> IO RawAstNode

foreign import ccall "isl/ast_build.h isl_ast_build_free"
    c_ast_build_free :: RawAstBuild -> IO ()

-- AST Node Operations
foreign import ccall "isl/ast.h isl_ast_node_get_type"
    c_ast_node_get_type :: RawAstNode -> IO CInt

foreign import ccall "isl/ast.h isl_ast_node_free"
    c_ast_node_free :: RawAstNode -> IO ()

foreign import ccall "isl/ast.h isl_ast_node_copy"
    c_ast_node_copy :: RawAstNode -> IO RawAstNode

-- AST Node For Operations
foreign import ccall "isl/ast.h isl_ast_node_for_get_iterator"
    c_ast_node_for_get_iterator :: RawAstNode -> IO RawAstExpr

foreign import ccall "isl/ast.h isl_ast_node_for_get_init"
    c_ast_node_for_get_init :: RawAstNode -> IO RawAstExpr

foreign import ccall "isl/ast.h isl_ast_node_for_get_cond"
    c_ast_node_for_get_cond :: RawAstNode -> IO RawAstExpr

foreign import ccall "isl/ast.h isl_ast_node_for_get_inc"
    c_ast_node_for_get_inc :: RawAstNode -> IO RawAstExpr

foreign import ccall "isl/ast.h isl_ast_node_for_get_body"
    c_ast_node_for_get_body :: RawAstNode -> IO RawAstNode

-- AST Node If Operations
foreign import ccall "isl/ast.h isl_ast_node_if_get_cond"
    c_ast_node_if_get_cond :: RawAstNode -> IO RawAstExpr

foreign import ccall "isl/ast.h isl_ast_node_if_get_then_node"
    c_ast_node_if_get_then_node :: RawAstNode -> IO RawAstNode

foreign import ccall "isl/ast.h isl_ast_node_if_has_else_node"
    c_ast_node_if_has_else_node :: RawAstNode -> IO CInt

foreign import ccall "isl/ast.h isl_ast_node_if_get_else_node"
    c_ast_node_if_get_else_node :: RawAstNode -> IO RawAstNode

-- AST Node Block Operations
foreign import ccall "isl/ast.h isl_ast_node_block_get_children"
    c_ast_node_block_get_children :: RawAstNode -> IO RawAstNodeList

-- AST Node User Operations
foreign import ccall "isl/ast.h isl_ast_node_user_get_expr"
    c_ast_node_user_get_expr :: RawAstNode -> IO RawAstExpr

-- AST Node Mark Operations
foreign import ccall "isl/ast.h isl_ast_node_mark_get_id"
    c_ast_node_mark_get_id :: RawAstNode -> IO RawId

foreign import ccall "isl/ast.h isl_ast_node_mark_get_node"
    c_ast_node_mark_get_node :: RawAstNode -> IO RawAstNode

-- AST Node List Operations
foreign import ccall "isl/ast.h isl_ast_node_list_n_ast_node"
    c_ast_node_list_n_ast_node :: RawAstNodeList -> IO CInt

foreign import ccall "isl/ast.h isl_ast_node_list_get_at"
    c_ast_node_list_get_at :: RawAstNodeList -> CInt -> IO RawAstNode

foreign import ccall "isl/ast.h isl_ast_node_list_free"
    c_ast_node_list_free :: RawAstNodeList -> IO ()

-- AST Expr Operations
foreign import ccall "isl/ast.h isl_ast_expr_get_type"
    c_ast_expr_get_type :: RawAstExpr -> IO CInt

foreign import ccall "isl/ast.h isl_ast_expr_free"
    c_ast_expr_free :: RawAstExpr -> IO ()

foreign import ccall "isl/ast.h isl_ast_expr_copy"
    c_ast_expr_copy :: RawAstExpr -> IO RawAstExpr

foreign import ccall "isl/ast.h isl_ast_expr_get_id"
    c_ast_expr_get_id :: RawAstExpr -> IO RawId

foreign import ccall "isl/ast.h isl_ast_expr_get_val"
    c_ast_expr_get_val :: RawAstExpr -> IO RawVal

foreign import ccall "isl/ast.h isl_ast_expr_get_op_type"
    c_ast_expr_get_op_type :: RawAstExpr -> IO CInt

foreign import ccall "isl/ast.h isl_ast_expr_get_op_n_arg"
    c_ast_expr_get_op_n_arg :: RawAstExpr -> IO CInt

foreign import ccall "isl/ast.h isl_ast_expr_get_op_arg"
    c_ast_expr_get_op_arg :: RawAstExpr -> CInt -> IO RawAstExpr

-- Val Operations
foreign import ccall "isl/val.h isl_val_get_num_si"
    c_val_get_num_si :: RawVal -> IO CLong

foreign import ccall "isl/val.h isl_val_free"
    c_val_free :: RawVal -> IO ()

-- Printer Operations
foreign import ccall "isl/printer.h isl_printer_to_str"
    c_printer_to_str :: RawCtx -> IO RawPrinter

foreign import ccall "isl/printer.h isl_printer_set_output_format"
    c_printer_set_output_format :: RawPrinter -> CInt -> IO RawPrinter

foreign import ccall "isl/ast.h isl_printer_print_ast_node"
    c_printer_print_ast_node :: RawPrinter -> RawAstNode -> IO RawPrinter

foreign import ccall "isl/printer.h isl_printer_get_str"
    c_printer_get_str :: RawPrinter -> IO CString

foreign import ccall "isl/printer.h isl_printer_free"
    c_printer_free :: RawPrinter -> IO ()

-- AST Node Type Constants (isl_ast_node_type enum)
-- From isl/ast_type.h
astNodeTypeError, astNodeTypeFor, astNodeTypeIf :: CInt
astNodeTypeBlock, astNodeTypeMark, astNodeTypeUser :: CInt
astNodeTypeError = -1
astNodeTypeFor = 1
astNodeTypeIf = 2

astNodeTypeBlock = 3

astNodeTypeMark = 4

astNodeTypeUser = 5

-- AST Expr Type Constants (isl_ast_expr_type enum)
-- From isl/ast_type.h
astExprTypeError, astExprTypeId, astExprTypeInt, astExprTypeOp :: CInt
astExprTypeError = -1
astExprTypeId = 1
astExprTypeInt = 2
astExprTypeOp = 0

-- AST Op Type Constants (isl_ast_expr_op_type enum)
-- From isl/ast_type.h
astOpError, astOpAnd, astOpAndThen, astOpOr, astOpOrElse :: CInt
astOpMax, astOpMin, astOpMinus, astOpAdd, astOpSub :: CInt
astOpMul, astOpDiv, astOpFdivQ, astOpPdivQ, astOpPdivR :: CInt
astOpZdivR, astOpCond, astOpSelect, astOpEq, astOpLe :: CInt
astOpLt, astOpGe, astOpGt, astOpCall, astOpAccess :: CInt
astOpMember, astOpAddressOf :: CInt
astOpError = -1
astOpAnd = 0
astOpAndThen = 1
astOpOr = 2
astOpOrElse = 3

astOpMax = 4

astOpMin = 5

astOpMinus = 6

astOpAdd = 7

astOpSub = 8

astOpMul = 9

astOpDiv = 10

astOpFdivQ = 11

astOpPdivQ = 12

astOpPdivR = 13

astOpZdivR = 14

astOpCond = 15

astOpSelect = 16

astOpEq = 17

astOpLe = 18

astOpLt = 19

astOpGe = 20

astOpGt = 21

astOpCall = 22

astOpAccess = 23

astOpMember = 24

astOpAddressOf = 25

-- Printer Format Constants
-- From isl/printer.h
formatIsl, formatC :: CInt
formatIsl = 0
formatC = 4
