module ISL.ScheduleTreeSpec (spec) where

import           Data.List  (sortOn)
import           ISL
import qualified ISL.Ast    as Ast
import           Test.Hspec

normalizeUnionSetString :: String -> Either String String
normalizeUnionSetString str = do
    Ast.UnionSetExpr sets <- Ast.parseUnionSetExpr str
    let sorted = sortOn Ast.setExprToString sets
    pure $ Ast.unionSetExprToString (Ast.UnionSetExpr sorted)

normalizeScheduleTree :: ScheduleTree -> Either String ScheduleTree
normalizeScheduleTree tree =
    case tree of
        TreeBand info children ->
            TreeBand info <$> traverse normalizeScheduleTree children
        TreeContext ctx children ->
            TreeContext ctx <$> traverse normalizeScheduleTree children
        TreeDomain dom children ->
            TreeDomain <$> normalizeUnionSetString dom <*> traverse normalizeScheduleTree children
        TreeFilter filt children ->
            TreeFilter <$> normalizeUnionSetString filt <*> traverse normalizeScheduleTree children
        TreeGuard guardStr children ->
            TreeGuard guardStr <$> traverse normalizeScheduleTree children
        TreeMark mark children ->
            TreeMark mark <$> traverse normalizeScheduleTree children
        TreeExtension ext children ->
            TreeExtension ext <$> traverse normalizeScheduleTree children
        TreeSequence children ->
            TreeSequence <$> traverse normalizeScheduleTree children
        TreeSet children ->
            TreeSet <$> traverse normalizeScheduleTree children
        TreeExpansion children ->
            TreeExpansion <$> traverse normalizeScheduleTree children
        TreeLeaf ->
            Right TreeLeaf
        TreeUnknown tag children ->
            TreeUnknown tag <$> traverse normalizeScheduleTree children

normalizeResult :: Either IslError ScheduleTree -> Either String ScheduleTree
normalizeResult = either (Left . show) normalizeScheduleTree

spec :: Spec
spec = do
    describe "ISL Schedule Tree" $ do
        it "can get schedule tree from domain-only schedule" $ do
            result <- runISL $ do
                dom <- unionSet "{ S[i] : 0 <= i < 10 }"
                sched <- scheduleFromDomain dom
                scheduleTree sched
            normalizeResult result
                `shouldBe` normalizeScheduleTree
                    ( TreeDomain
                        "{ S[i] : 0 <= i <= 9 }"
                        [TreeLeaf]
                    )

        it "can get schedule tree with band node" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                scheduleTree sched
            normalizeResult result
                `shouldBe` normalizeScheduleTree
                    ( TreeDomain
                        "{ S[i] : 0 <= i <= 9 }"
                        [ TreeBand
                            BandInfo
                                { bandSchedule = "[{ S[i] -> [(i)] }]"
                                , bandPermutable = False
                                , bandMembers = 1
                                }
                            [TreeLeaf]
                        ]
                    )

        it "can get schedule tree with sequence node" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ A[i] : 0 <= i < 5; B[j] : 0 <= j < 3 }\", child: { sequence: [ { filter: \"{ A[i] }\" }, { filter: \"{ B[j] }\" } ] } }"
                scheduleTree sched
            normalizeResult result
                `shouldBe` normalizeScheduleTree
                    ( TreeDomain
                        "{ A[i] : 0 <= i <= 4; B[j] : 0 <= j <= 2 }"
                        [ TreeSequence
                            [ TreeFilter "{ A[i] }" [TreeLeaf]
                            , TreeFilter "{ B[j] }" [TreeLeaf]
                            ]
                        ]
                    )
