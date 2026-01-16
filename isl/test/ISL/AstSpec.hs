module ISL.AstSpec (spec) where

import           ISL
import           ISL.Ast
import           Test.Hspec

spec :: Spec
spec = do
    describe "ISL AST Build" $ do
        it "can allocate AST build" $ do
            result <- runISL $ do
                _ <- astBuildAlloc
                pure ()
            result `shouldBe` Right ()

        it "can create AST build from context" $ do
            result <- runISL $ do
                ctx <- set "{ : }"
                _ <- astBuildFromContext ctx
                pure ()
            result `shouldBe` Right ()

    describe "ISL AST Generation" $ do
        it "can generate AST from simple schedule" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToTree node
            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        AstFor{forIterator = iter, forBody = body} -> do
                            iter `shouldBe` "c0"
                            case body of
                                AstUser _ -> pure ()
                                _         -> expectationFailure $ "Expected AstUser, got: " ++ show body
                        _ -> expectationFailure $ "Expected AstFor, got: " ++ show tree

        it "can generate C code from schedule" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToC node
            case result of
                Left err -> expectationFailure $ show err
                Right cCode -> do
                    cCode `shouldContain` "for"
                    cCode `shouldContain` "c0"

        it "can generate AST with nested loops" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i, j] : 0 <= i < 4 and 0 <= j < 4 }\", child: { schedule: \"[{ S[i, j] -> [(i)]; S[i, j] -> [(j)] }]\" } }"
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToTree node
            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        AstFor{forIterator = outerIter, forBody = innerLoop} -> do
                            outerIter `shouldBe` "c0"
                            case innerLoop of
                                AstFor{forIterator = innerIter} ->
                                    innerIter `shouldBe` "c1"
                                _ -> expectationFailure $ "Expected inner AstFor, got: " ++ show innerLoop
                        _ -> expectationFailure $ "Expected AstFor, got: " ++ show tree

    describe "ISL AST Expression" $ do
        it "can extract expressions from for loop" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [i] }]\" } }"
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToTree node
            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        AstFor{forInit = initExpr, forCond = condExpr, forInc = incExpr} -> do
                            initExpr `shouldBe` ExprInt 0
                            incExpr `shouldBe` ExprInt 1
                            case condExpr of
                                ExprOp OpLe [ExprId "c0", ExprInt 9] -> pure ()
                                _ -> expectationFailure $ "Unexpected condition: " ++ show condExpr
                        _ -> expectationFailure $ "Expected AstFor, got: " ++ show tree

    describe "ISL AST Operations" $ do
        it "can handle arithmetic expressions" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ S[i] : 0 <= i < 10 }\", child: { schedule: \"[{ S[i] -> [2*i + 1] }]\" } }"
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToC node
            case result of
                Left err -> expectationFailure $ show err
                Right cCode -> do
                    cCode `shouldContain` "for"

    describe "ISL AST Block" $ do
        it "can generate AST with sequence (block)" $ do
            result <- runISL $ do
                sched <-
                    schedule
                        "{ domain: \"{ A[i] : 0 <= i < 5; B[j] : 0 <= j < 3 }\", child: { sequence: [ { filter: \"{ A[i] }\", child: { schedule: \"[{ A[i] -> [i] }]\" } }, { filter: \"{ B[j] }\", child: { schedule: \"[{ B[j] -> [j] }]\" } } ] } }"
                build <- astBuildAlloc
                node <- astBuildNodeFromSchedule build sched
                astNodeToTree node
            case result of
                Left err -> expectationFailure $ show err
                Right tree -> do
                    case tree of
                        AstBlock nodes -> length nodes `shouldBe` 2
                        _ -> expectationFailure $ "Expected AstBlock, got: " ++ show tree
