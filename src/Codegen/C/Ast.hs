{-# LANGUAGE OverloadedRecordDot #-}

module Codegen.C.Ast (
    CAstError (..),
    CFunctionName (..),
    CTensorName (..),
    CExpr (..),
    CTensorRef (..),
    CStmt (..),
    CProgram (..),
    lowerToCProgram,
) where

import           Codegen.GenIR    (AstIterVar, ExtentParamName (..),
                                   GenExpr (..), GenProgram (..), GenStmt (..),
                                   GenTensorDecl (..), GenTensorRef (..))
import qualified Data.Map.Strict  as Map
import qualified Data.Set         as Set
import           Data.String      (IsString (fromString))
import           FrontendIR.Types (TensorName, tensorNameToString)

data CAstError
    = ErrCAstReductionNotSupported
    | ErrCAstUnknownTensorShape TensorName
    | ErrCAstTensorRankMismatch TensorName Int Int
    deriving (Eq, Show)

newtype CFunctionName = CFunctionName String deriving (Eq, Ord, Show)

instance IsString CFunctionName where fromString = CFunctionName

newtype CTensorName = CTensorName String deriving (Eq, Ord, Show)

instance IsString CTensorName where fromString = CTensorName

data CExpr
    = CVar AstIterVar
    | CInt Int
    | CLoad CTensorName [CExpr]
    | CAdd CExpr CExpr
    | CMul CExpr CExpr
    deriving (Eq, Show)

data CTensorRef = CTensorRef
    { tensorName :: CTensorName
    , tensorIndices :: [CExpr]
    }
    deriving (Eq, Show)

data CStmt
    = CForLoop
        { cForVar :: AstIterVar
        , cForUpperBound :: ExtentParamName
        , cForBody :: [CStmt]
        }
    | CAssign
        { cAssignTarget :: CTensorRef
        , cAssignExpr :: CExpr
        }
    deriving (Eq, Show)

data CProgram = CProgram
    { cFunctionName :: CFunctionName
    , cExtentParams :: [ExtentParamName]
    , cTensorArgs :: [CTensorName]
    , cBody :: [CStmt]
    }
    deriving (Eq, Show)

lowerToCProgram :: GenProgram -> Either CAstError CProgram
lowerToCProgram genProgram = do
    cBody <- lowerStmts tensorShapes genProgram.genBody
    pure
        CProgram
            { cFunctionName = fromString "shoto_kernel"
            , cExtentParams = genProgram.genExtentParams
            , cTensorArgs = collectTensorArgs genProgram.genBody
            , cBody
            }
  where
    tensorShapes = buildTensorShapeMap genProgram.genTensorDecls

lowerStmts ::
    Map.Map TensorName [ExtentParamName] ->
    [GenStmt] ->
    Either CAstError [CStmt]
lowerStmts tensorShapes =
    traverse (lowerStmt tensorShapes)

lowerStmt ::
    Map.Map TensorName [ExtentParamName] ->
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
    Map.Map TensorName [ExtentParamName] ->
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
    Map.Map TensorName [ExtentParamName] ->
    GenTensorRef ->
    Either CAstError CTensorRef
lowerTensorRef tensorShapes ref = do
    shapeParams <- lookupTensorShape tensorShapes ref.genTensor
    flatIndex <- flattenRowMajor ref.genTensor (CVar <$> ref.genIndices) shapeParams
    pure
        CTensorRef
            { tensorName = fromString $ tensorNameToString ref.genTensor
            , tensorIndices = [flatIndex]
            }

lookupTensorShape ::
    Map.Map TensorName [ExtentParamName] ->
    TensorName ->
    Either CAstError [ExtentParamName]
lookupTensorShape tensorShapes tensor =
    case Map.lookup tensor tensorShapes of
        Just shapeParams -> pure shapeParams
        Nothing -> Left $ ErrCAstUnknownTensorShape tensor

flattenRowMajor ::
    TensorName ->
    [CExpr] ->
    [ExtentParamName] ->
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
            (CMul acc (CVar (fromString (extentParamName extentParam))))
            indexExpr

    extentParamName (ExtentParamName name) = name

buildTensorShapeMap :: [GenTensorDecl] -> Map.Map TensorName [ExtentParamName]
buildTensorShapeMap tensorDecls =
    Map.fromList
        [ (tensorDecl.genTensor, tensorDecl.genShape)
        | tensorDecl <- tensorDecls
        ]

collectTensorArgs :: [GenStmt] -> [CTensorName]
collectTensorArgs stmts =
    uniqueStable $ concatMap collectStmtTensors stmts
  where
    collectStmtTensors stmt =
        case stmt of
            GenFor{} -> collectTensorArgs stmt.genBody
            GenAssign{} ->
                fromString (tensorNameToString stmt.genTarget.genTensor)
                    : collectExprTensors stmt.genExpr
            GenReduction{} ->
                fromString (tensorNameToString stmt.genTarget.genTensor)
                    : collectExprTensors stmt.genExpr

    collectExprTensors expr =
        case expr of
            GenConst _ -> []
            GenLoad ref -> [fromString $ tensorNameToString ref.genTensor]
            GenAdd lhs rhs -> collectExprTensors lhs <> collectExprTensors rhs
            GenMul lhs rhs -> collectExprTensors lhs <> collectExprTensors rhs

uniqueStable :: (Ord a) => [a] -> [a]
uniqueStable = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
        | x `Set.member` seen = go seen xs
        | otherwise = x : go (Set.insert x seen) xs
