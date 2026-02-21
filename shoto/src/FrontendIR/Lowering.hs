{-# LANGUAGE OverloadedRecordDot #-}

module FrontendIR.Lowering (
    CheckedProgram,
    checkProgram,
    lowerToRaw,
) where

import           Data.List          (intercalate)
import qualified Data.List.NonEmpty as NE
import qualified Data.Map.Strict    as Map
import qualified Data.Set           as Set
import           FrontendIR.Types   (Axis (..), Expr (..), FrontendError (..),
                                     IterName, IxExpr (..), ParamName,
                                     Program (..), Stmt (..), TensorDecl (..),
                                     TensorName, iterNameToString,
                                     paramNameToString, tensorNameToString)
import           Polyhedral.Parse   (RawPolyhedralModel (..))

newtype CheckedProgram = UnsafeMkCheckedProgram Program

lowerToRaw :: CheckedProgram -> RawPolyhedralModel
lowerToRaw (UnsafeMkCheckedProgram Program{axes = axisList, stmt = statement}) =
    let params = (.extent) <$> NE.toList axisList
        iters = (.iter) <$> NE.toList axisList
     in RawPolyhedralModel
            { context = mkContext params
            , domain = mkDomain params iters
            , programOrder = mkProgramOrder params iters
            , readAccess = mkReadAccess params iters (collectLoads statement.rhs)
            , writeAccess = mkWriteAccess params iters statement
            , reductionDomain = "{ }"
            , reductionRead = "{ }"
            , reductionWrite = "{ }"
            }

checkProgram :: Program -> Either FrontendError CheckedProgram
checkProgram p@Program{axes = axisList, tensors = tensorDecls, stmt = statement} = do
    let axesList = NE.toList axisList
    let tensorList = NE.toList tensorDecls
    let expectedIters = (.iter) <$> axesList
    let knownShapeParams = Set.fromList ((.extent) <$> axesList)
    let tensorRanks = Map.fromList ((\decl -> (decl.tensor, length decl.shape)) <$> tensorList)
    maybe (Right ()) (Left . ErrDuplicateIter) (firstDuplicate ((.iter) <$> axesList))
    maybe (Right ()) (Left . ErrDuplicateParam) (firstDuplicate ((.extent) <$> axesList))
    maybe (Right ()) (Left . ErrDuplicateTensor) (firstDuplicate ((.tensor) <$> tensorList))
    ensureShapeParamsKnown knownShapeParams tensorList
    let actualStore = ixExprName <$> statement.outputIndex
    if actualStore == expectedIters
        then Right ()
        else Left $ ErrStoreIndexMismatch expectedIters actualStore
    ensureTensorDeclared tensorRanks statement.outputTensor
    ensureRankMatches tensorRanks statement.outputTensor statement.outputIndex
    checkExpr expectedIters tensorRanks statement.rhs
    pure (UnsafeMkCheckedProgram p)
  where
    ensureShapeParamsKnown :: Set.Set ParamName -> [TensorDecl] -> Either FrontendError ()
    ensureShapeParamsKnown _ [] = Right ()
    ensureShapeParamsKnown knownParams (decl : rest) = do
        ensureShapeParamsKnown' decl.tensor knownParams decl.shape
        ensureShapeParamsKnown knownParams rest

    ensureShapeParamsKnown' :: TensorName -> Set.Set ParamName -> [ParamName] -> Either FrontendError ()
    ensureShapeParamsKnown' _ _ [] = Right ()
    ensureShapeParamsKnown' tensorName knownParams (param : params)
        | Set.member param knownParams = ensureShapeParamsKnown' tensorName knownParams params
        | otherwise = Left $ ErrUnknownTensorShapeParam tensorName param

    ensureTensorDeclared :: Map.Map TensorName Int -> TensorName -> Either FrontendError ()
    ensureTensorDeclared tensorRankMap tensorName =
        case Map.lookup tensorName tensorRankMap of
            Nothing -> Left $ ErrUndeclaredTensor tensorName
            Just _ -> Right ()

    ensureRankMatches :: Map.Map TensorName Int -> TensorName -> [IxExpr] -> Either FrontendError ()
    ensureRankMatches tensorRankMap tensorName indices =
        case Map.lookup tensorName tensorRankMap of
            Nothing -> Left $ ErrUndeclaredTensor tensorName
            Just expectedRank ->
                if actualRank == expectedRank
                    then Right ()
                    else Left $ ErrTensorRankMismatch tensorName expectedRank actualRank
      where
        actualRank = length indices

    checkExpr :: [IterName] -> Map.Map TensorName Int -> Expr -> Either FrontendError ()
    checkExpr _ _ (EConst _) = Right ()
    checkExpr expected tensorRankMap (EAdd lhs rhs) =
        checkExpr expected tensorRankMap lhs >> checkExpr expected tensorRankMap rhs
    checkExpr expected tensorRankMap (EMul lhs rhs) =
        checkExpr expected tensorRankMap lhs >> checkExpr expected tensorRankMap rhs
    checkExpr expected tensorRankMap (ELoad tensorName indices) = do
        ensureTensorDeclared tensorRankMap tensorName
        ensureRankMatches tensorRankMap tensorName indices
        if actual == expected
            then Right ()
            else Left $ ErrLoadIndexMismatch tensorName expected actual
      where
        actual = ixExprName <$> indices

firstDuplicate :: (Ord a) => [a] -> Maybe a
firstDuplicate = go Set.empty
  where
    go _ [] = Nothing
    go seen (x : xs)
        | Set.member x seen = Just x
        | otherwise = go (Set.insert x seen) xs

mkContext :: [ParamName] -> String
mkContext params
    | null params = "{ : }"
    | otherwise = withParams params $ "{ : " <> constraints <> " }"
  where
    constraints = intercalate " and " (mkNonNegative <$> params)

mkDomain :: [ParamName] -> [IterName] -> String
mkDomain params iters = withOptionalParams params body
  where
    body = "{ " <> statementRef iters <> " : " <> domainConstraints <> " }"
    domainConstraints = intercalate " and " (mkRange <$> zip iters params)

mkProgramOrder :: [ParamName] -> [IterName] -> String
mkProgramOrder params iters =
    withOptionalParams params $
        "{ " <> statementRef iters <> " -> " <> tuple iterNameToString iters <> " }"

mkReadAccess :: [ParamName] -> [IterName] -> [(TensorName, [IxExpr])] -> String
mkReadAccess _ _ [] = "{ }"
mkReadAccess params iters loads =
    withOptionalParams params $
        "{ " <> intercalate "; " relations <> " }"
  where
    relations =
        uniqueStable $
            (\(tensor, indices) -> statementRef iters <> " -> " <> tensorRef tensor indices) <$> loads

mkWriteAccess :: [ParamName] -> [IterName] -> Stmt -> String
mkWriteAccess params iters statement =
    withOptionalParams params $
        "{ "
            <> statementRef iters
            <> " -> "
            <> tensorRef statement.outputTensor statement.outputIndex
            <> " }"

collectLoads :: Expr -> [(TensorName, [IxExpr])]
collectLoads (EConst _) = []
collectLoads (ELoad tensor indices) = [(tensor, indices)]
collectLoads (EAdd lhs rhs) = collectLoads lhs <> collectLoads rhs
collectLoads (EMul lhs rhs) = collectLoads lhs <> collectLoads rhs

withOptionalParams :: [ParamName] -> String -> String
withOptionalParams [] body = body
withOptionalParams params body = withParams params body

withParams :: [ParamName] -> String -> String
withParams params body = tuple paramNameToString params <> " -> " <> body

tuple :: (a -> String) -> [a] -> String
tuple render xs = "[" <> intercalate "," (render <$> xs) <> "]"

statementRef :: [IterName] -> String
statementRef iters = "S" <> tuple iterNameToString iters

tensorRef :: TensorName -> [IxExpr] -> String
tensorRef tensor indices = tensorNameToString tensor <> tuple iterNameToString (ixExprName <$> indices)

ixExprName :: IxExpr -> IterName
ixExprName (IxVar name) = name

mkNonNegative :: ParamName -> String
mkNonNegative param = "0 <= " <> paramNameToString param

mkRange :: (IterName, ParamName) -> String
mkRange (iter, extent) =
    "0 <= " <> iterNameToString iter <> " < " <> paramNameToString extent

uniqueStable :: (Ord a) => [a] -> [a]
uniqueStable = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
        | Set.member x seen = go seen xs
        | otherwise = x : go (Set.insert x seen) xs
