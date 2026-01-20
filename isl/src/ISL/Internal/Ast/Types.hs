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
    = OpAnd
    | OpAndThen
    | OpOr
    | OpOrElse
    | OpMax
    | OpMin
    | OpMinus
    | OpAdd
    | OpSub
    | OpMul
    | OpDiv
    | OpFdivQ
    | OpPdivQ
    | OpPdivR
    | OpZdivR
    | OpCond
    | OpSelect
    | OpEq
    | OpLe
    | OpLt
    | OpGe
    | OpGt
    | OpCall
    | OpAccess
    | OpMember
    | OpAddressOf
    | OpUnknown Int
    deriving (Show, Eq)

-- | Pure Haskell representation of AST expressions
data AstExpression
    = ExprId String
    | ExprInt Integer
    | ExprOp AstOp [AstExpression]
    | ExprError
    deriving (Show, Eq)

-- | Pure Haskell representation of AST nodes
data AstTree
    = AstFor
        { forIterator :: String
        , forInit :: AstExpression
        , forCond :: AstExpression
        , forInc :: AstExpression
        , forBody :: AstTree
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
