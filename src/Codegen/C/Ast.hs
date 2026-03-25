{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.C.Ast (
    CAstError (..),
    CExpr (..),
    CStmt (..),
    CProgram (..),
    lowerToCProgram,
) where

import           Codegen.GenIR    (GenExpr (..), GenProgram (..), GenStmt (..))
import qualified Data.Map.Strict  as Map
import qualified Data.Set         as Set
import           FrontendIR.Types (IterName, ParamName, TensorName)
import           IR.Name          (KernelName (..))
import           IR.Tensor        (TensorDecl (..), TensorRef (..))

data CAstError
    = ErrCAstReductionNotSupported
    | ErrCAstUnknownTensorShape TensorName
    | ErrCAstTensorRankMismatch TensorName Int Int
    deriving (Eq, Show)

data CExpr
    = CVar IterName
    | CExtentVar ParamName
    | CInt Int
    | CLoad TensorName [CExpr]
    | CAdd CExpr CExpr
    | CMul CExpr CExpr
    deriving (Eq, Show)

data CStmt
    = CForLoop
        { cForVar :: IterName
        , cForUpperBound :: ParamName
        , cForBody :: [CStmt]
        }
    | CAssign
        { cAssignTarget :: TensorRef CExpr
        , cAssignExpr :: CExpr
        }
    deriving (Eq, Show)

data CProgram = CProgram
    { cFunctionName :: KernelName
    , cExtentParams :: [ParamName]
    , cTensorArgs :: [TensorName]
    , cBody :: [CStmt]
    }
    deriving (Eq, Show)

lowerToCProgram :: GenProgram -> Either CAstError CProgram
lowerToCProgram genProgram = do
    cBody <- lowerStmts tensorShapes genProgram.genBody
    pure
        CProgram
            { cFunctionName = KernelName "shoto_kernel"
            , cExtentParams = genProgram.genExtentParams
            , cTensorArgs = collectTensorArgs genProgram.genBody
            , cBody
            }
  where
    tensorShapes = buildTensorShapeMap genProgram.genTensorDecls

lowerStmts ::
    Map.Map TensorName [ParamName] ->
    [GenStmt] ->
    Either CAstError [CStmt]
lowerStmts tensorShapes =
    traverse (lowerStmt tensorShapes)

lowerStmt ::
    Map.Map TensorName [ParamName] ->
    GenStmt ->
    Either CAstError CStmt
lowerStmt tensorShapes stmt =
    case stmt of
        GenFor{} ->
            CForLoop
                <$> pure stmt.genIter
                <*> pure stmt.genBound
                <*> lowerStmts tensorShapes stmt.genBody
        GenAssign{} ->
            CAssign
                <$> lowerTensorRef tensorShapes stmt.genTarget
                <*> lowerExpr tensorShapes stmt.genExpr
        GenReduction{} -> Left ErrCAstReductionNotSupported

lowerExpr ::
    Map.Map TensorName [ParamName] ->
    GenExpr ->
    Either CAstError CExpr
lowerExpr tensorShapes expr =
    case expr of
        GenConst value -> pure $ CInt value
        GenLoad ref -> do
            loweredRef <- lowerTensorRef tensorShapes ref
            pure $ CLoad loweredRef.tensorName loweredRef.tensorIndices
        GenAdd lhs rhs -> CAdd <$> lowerExpr tensorShapes lhs <*> lowerExpr tensorShapes rhs
        GenMul lhs rhs -> CMul <$> lowerExpr tensorShapes lhs <*> lowerExpr tensorShapes rhs

lowerTensorRef ::
    Map.Map TensorName [ParamName] ->
    TensorRef IterName ->
    Either CAstError (TensorRef CExpr)
lowerTensorRef tensorShapes ref = do
    shapeParams <- lookupTensorShape tensorShapes ref.tensorName
    flatIndex <- flattenRowMajor ref.tensorName (CVar <$> ref.tensorIndices) shapeParams
    pure
        TensorRef
            { tensorName = ref.tensorName
            , tensorIndices = [flatIndex]
            }

lookupTensorShape ::
    Map.Map TensorName [ParamName] ->
    TensorName ->
    Either CAstError [ParamName]
lookupTensorShape tensorShapes tensor =
    case Map.lookup tensor tensorShapes of
        Just shapeParams -> pure shapeParams
        Nothing -> Left $ ErrCAstUnknownTensorShape tensor

flattenRowMajor ::
    TensorName ->
    [CExpr] ->
    [ParamName] ->
    Either CAstError CExpr
flattenRowMajor tensor loweredIndices shapeParams
    | length loweredIndices /= length shapeParams =
        Left $
            ErrCAstTensorRankMismatch
                tensor
                (length shapeParams)
                (length loweredIndices)
    | otherwise =
        case (loweredIndices, shapeParams) of
            ([], []) -> pure $ CInt 0
            (firstIndex : restIndices, _ : restShapeParams) ->
                pure $ foldl step firstIndex (zip restIndices restShapeParams)
            _ -> Left $ ErrCAstTensorRankMismatch tensor (length shapeParams) (length loweredIndices)
  where
    step acc (indexExpr, extentParam) =
        CAdd
            (CMul acc (CExtentVar extentParam))
            indexExpr

buildTensorShapeMap :: [TensorDecl] -> Map.Map TensorName [ParamName]
buildTensorShapeMap tensorDecls =
    Map.fromList
        [ (tensorDecl.tensor, tensorDecl.shape)
        | tensorDecl <- tensorDecls
        ]

collectTensorArgs :: [GenStmt] -> [TensorName]
collectTensorArgs stmts =
    uniqueStable $ concatMap collectStmtTensors stmts
  where
    collectStmtTensors stmt =
        case stmt of
            GenFor{} -> collectTensorArgs stmt.genBody
            GenAssign{} ->
                stmt.genTarget.tensorName : collectExprTensors stmt.genExpr
            GenReduction{} ->
                stmt.genTarget.tensorName : collectExprTensors stmt.genExpr

    collectExprTensors expr =
        case expr of
            GenConst _ -> []
            GenLoad ref -> [ref.tensorName]
            GenAdd lhs rhs -> collectExprTensors lhs <> collectExprTensors rhs
            GenMul lhs rhs -> collectExprTensors lhs <> collectExprTensors rhs

uniqueStable :: (Ord a) => [a] -> [a]
uniqueStable = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
        | x `Set.member` seen = go seen xs
        | otherwise = x : go (Set.insert x seen) xs
