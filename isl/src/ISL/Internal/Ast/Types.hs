module ISL.Internal.Ast.Types (
    -- * ISL Wrapper Types
    AstBuild (..),
    AstNode (..),
    AstExpr (..),

    -- * Pure Haskell AST Types
    AstTree (..),
    AstExpression (..),
    AstOp (..),
) where

import           Foreign.ForeignPtr (ForeignPtr)
import           ISL.Internal.FFI   (IslAstBuild, IslAstExpr, IslAstNode)

-- | ISL AST Build wrapper
newtype AstBuild s = AstBuild (ForeignPtr IslAstBuild)

-- | ISL AST Node wrapper
newtype AstNode s = AstNode (ForeignPtr IslAstNode)

-- | ISL AST Expression wrapper
newtype AstExpr s = AstExpr (ForeignPtr IslAstExpr)

-- | Pure Haskell representation of AST operations
data AstOp
    = OpAnd [AstExpression]
    | OpAndThen [AstExpression]
    | OpOr [AstExpression]
    | OpOrElse [AstExpression]
    | OpMax [AstExpression]
    | OpMin [AstExpression]
    | OpMinus AstExpression
    | OpAdd AstExpression AstExpression
    | OpSub AstExpression AstExpression
    | OpMul AstExpression AstExpression
    | OpDiv AstExpression AstExpression
    | OpFdivQ AstExpression AstExpression
    | OpPdivQ AstExpression AstExpression
    | OpPdivR AstExpression AstExpression
    | OpZdivR AstExpression AstExpression
    | OpCond AstExpression AstExpression AstExpression
    | OpSelect AstExpression AstExpression AstExpression
    | OpEq AstExpression AstExpression
    | OpLe AstExpression AstExpression
    | OpLt AstExpression AstExpression
    | OpGe AstExpression AstExpression
    | OpGt AstExpression AstExpression
    | OpCall AstExpression [AstExpression]
    | OpAccess AstExpression [AstExpression]
    | OpMember AstExpression AstExpression
    | OpAddressOf AstExpression
    | OpUnknown Int [AstExpression]
    deriving (Show, Eq)

-- | Pure Haskell representation of AST expressions
data AstExpression
    = ExprId String
    | ExprInt Integer
    | ExprOp AstOp
    | ExprError
    deriving (Show, Eq)

-- | Pure Haskell representation of AST nodes
data AstTree
    = AstFor
        { forIterator :: String
        , forInit     :: AstExpression
        , forCond     :: AstExpression
        , forInc      :: AstExpression
        , forBody     :: AstTree
        }
    | AstIf
        { ifCond :: AstExpression
        , ifThen :: AstTree
        , ifElse :: Maybe AstTree
        }
    | AstBlock [AstTree]
    | AstUser AstExpression
    | AstMark String AstTree
    | AstError
    deriving (Show, Eq)
