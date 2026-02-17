module FrontendIR.Lowering (
    lowerToRaw,
) where

import           Data.List        (intercalate)
import qualified Data.Set         as Set
import           FrontendIR.Types (Axis (..), Expr (..), FrontendError (..),
                                   IterName, IxExpr (..), ParamName,
                                   Program (..), Stmt (..), TensorName,
                                   iterNameToString, paramNameToString,
                                   tensorNameToString)
import           Polyhedral.Parse (RawPolyhedralModel (..))

lowerToRaw :: Program -> Either FrontendError RawPolyhedralModel
lowerToRaw p@Program{axes = axisList, stmt = statement} = do
    validateProgram p
    let params = extent <$> axisList
    let iters = iter <$> axisList
    pure
        RawPolyhedralModel
            { context = mkContext params
            , domain = mkDomain params iters
            , programOrder = mkProgramOrder params iters
            , readAccess = mkReadAccess params iters (collectLoads (rhs statement))
            , writeAccess = mkWriteAccess params iters statement
            , reductionDomain = "{ }"
            , reductionRead = "{ }"
            , reductionWrite = "{ }"
            }

validateProgram :: Program -> Either FrontendError ()
validateProgram Program{axes = axisList, stmt = statement} = do
    if null axisList then Left ErrNoAxis else Right ()
    maybe (Right ()) (Left . ErrDuplicateIter) (firstDuplicate (iter <$> axisList))
    maybe (Right ()) (Left . ErrDuplicateParam) (firstDuplicate (extent <$> axisList))
    let expected = iter <$> axisList
    let actualStore = ixExprName <$> outputIndex statement
    if actualStore == expected
        then Right ()
        else Left $ ErrStoreIndexMismatch expected actualStore
    validateExpr expected (rhs statement)

validateExpr :: [IterName] -> Expr -> Either FrontendError ()
validateExpr _ (EConst _) = Right ()
validateExpr expected (EAdd lhs rhs) = validateExpr expected lhs >> validateExpr expected rhs
validateExpr expected (EMul lhs rhs) = validateExpr expected lhs >> validateExpr expected rhs
validateExpr expected (ELoad tensor indices) =
    if actual == expected
        then Right ()
        else Left $ ErrLoadIndexMismatch tensor expected actual
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
            <> tensorRef (outputTensor statement) (outputIndex statement)
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
