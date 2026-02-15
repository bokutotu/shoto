module ISL.ScheduleNodeSpec (spec) where

import           ISL
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL ScheduleNode" $ do
        it "can roundtrip schedule -> root -> schedule" $ do
            result <- runISL $ do
                sched <- schedule schedule2D
                root <- scheduleGetRoot sched
                sched' <- scheduleNodeGetSchedule root
                scheduleIsEqual sched sched'
            result `shouldBe` Right True

        it "can insert a mark above a band node (via bottom-up traversal)" $ do
            result <- runISL $ do
                sched <- schedule schedule2D
                sched' <-
                    scheduleMapScheduleNodeBottomUp sched $ \node -> do
                        ty <- scheduleNodeGetType node
                        case ty of
                            ScheduleNodeBand -> scheduleNodeInsertMark "BAND" node
                            _ -> pure node
                marks <- scheduleFindMarks "BAND" sched'
                pure (length marks)
            result `shouldBe` Right 1

        it "can permute band members (swap axis) on a schedule" $ do
            result <- runISL $ do
                sched <- schedule schedule2D
                sched' <-
                    scheduleMapScheduleNodeBottomUp sched $ \node -> do
                        ty <- scheduleNodeGetType node
                        case ty of
                            ScheduleNodeBand -> do
                                n <- scheduleNodeBandNMember node
                                if n == 2 then scheduleNodeBandPermuteMembers [1, 0] node else pure node
                            _ -> pure node
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched'
                astNodeToTree node

            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        AstFor{forBody = inner} ->
                            case inner of
                                AstFor{forBody = AstUser expr} -> do
                                    expr
                                        `shouldBe` ExprOp (OpCall (ExprId "S") [ExprId "c1", ExprId "c0"])
                                _ -> expectationFailure $ "Expected inner AstFor with AstUser, got: " ++ show inner
                        _ -> expectationFailure $ "Expected outer AstFor, got: " ++ show tree

        it "band tiling increases the number of band nodes" $ do
            result <- runISL $ do
                sched <- schedule schedule2D
                n0 <- countBandNodes sched
                sched' <-
                    scheduleMapScheduleNodeBottomUp sched $ \node -> do
                        ty <- scheduleNodeGetType node
                        case ty of
                            ScheduleNodeBand -> do
                                n <- scheduleNodeBandNMember node
                                if n == 2 then scheduleNodeBandTile [2, 2] node else pure node
                            _ -> pure node
                n1 <- countBandNodes sched'
                pure (n0, n1)
            result `shouldBe` Right (1, 2)

schedule2D :: String
schedule2D =
    "{ domain: \"{ S[i, j] : 0 <= i < 4 and 0 <= j < 4 }\", child: { schedule: \"[{ S[i, j] -> [(i)] }, { S[i, j] -> [(j)] }]\" } }"

countBandNodes :: Schedule s -> ISL s Int
countBandNodes sched = scheduleGetRoot sched >>= go
  where
    go node = do
        ty <- scheduleNodeGetType node
        let here =
                case ty of
                    ScheduleNodeBand -> 1
                    _ -> 0
        n <- scheduleNodeNChildren node
        children <- mapM (\i -> scheduleNodeChild node i >>= go) [0 .. n - 1]
        pure $ here + sum children
