module Polyhedral.OptimizeSpec (spec) where

import           Control.Monad       (void)
import           Data.Either         (isLeft)
import           ISL                 (runISL, schedule, scheduleIsEqual)
import           Polyhedral.Optimize (ScheduleOptimization (..),
                                      applyScheduleOptimizations)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Polyhedral schedule optimizations" $ do
        it "loop interchange can rotate 3D axes" $ do
            let inputSchedule =
                    unlines
                        [ "    |{ domain: \"{ S[i, j, k] : 0 <= i < 16 and 0 <= j < 16 and 0 <= k < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j, k] -> [(i)] }, { S[i, j, k] -> [(j)] }, { S[i, j, k] -> [(k)] }]\""
                        , "    |  }"
                        , "    |}"
                        ]
                expectedSchedule =
                    unlines
                        [ "    |{ domain: \"{ S[i, j, k] : 0 <= i < 16 and 0 <= j < 16 and 0 <= k < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j, k] -> [(j)] }, { S[i, j, k] -> [(k)] }, { S[i, j, k] -> [(i)] }]\""
                        , "    |  }"
                        , "    |}"
                        ]

            result <- runISL $ do
                sched <- schedule $ normalizePrettySchedule inputSchedule
                sched' <- applyScheduleOptimizations [LoopInterchange [1, 2, 0]] sched
                expected <- schedule $ normalizePrettySchedule expectedSchedule
                scheduleIsEqual sched' expected
            result `shouldBe` Right True

        it "tile supports multi-level tiling for all axes at once" $ do
            let inputSchedule =
                    unlines
                        [ "    |{ domain: \"{ S[i, j, k] : 0 <= i < 16 and 0 <= j < 16 and 0 <= k < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j, k] -> [(i)] }, { S[i, j, k] -> [(j)] }, { S[i, j, k] -> [(k)] }]\""
                        , "    |  }"
                        , "    |}"
                        ]
                expectedSchedule =
                    unlines
                        [ "    |{ domain: \"{ S[i, j, k] : 0 <= i < 16 and 0 <= j < 16 and 0 <= k < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j, k] -> [(i - (i) mod 4)] }, { S[i, j, k] -> [(j - (j) mod 4)] }, { S[i, j, k] -> [(k - (k) mod 4)] }]\""
                        , "    |  , child: {"
                        , "    |      schedule: \"[{ S[i, j, k] -> [(-1*((i) mod 2) + (i) mod 4)] }, { S[i, j, k] -> [(-1*((j) mod 2) + (j) mod 4)] }, { S[i, j, k] -> [(-1*((k) mod 2) + (k) mod 4)] }]\""
                        , "    |    , child: {"
                        , "    |        schedule: \"[{ S[i, j, k] -> [((i) mod 2)] }, { S[i, j, k] -> [((j) mod 2)] }, { S[i, j, k] -> [((k) mod 2)] }]\""
                        , "    |      }"
                        , "    |    }"
                        , "    |  }"
                        , "    |}"
                        ]

            result <- runISL $ do
                sched <- schedule $ normalizePrettySchedule inputSchedule
                sched' <- applyScheduleOptimizations [Tile [[4, 2], [4, 2], [4, 2]]] sched
                expected <- schedule $ normalizePrettySchedule expectedSchedule
                scheduleIsEqual sched' expected
            result `shouldBe` Right True

        it "optimization order affects resulting schedule" $ do
            let inputSchedule =
                    unlines
                        [ "    |{ domain: \"{ S[i, j] : 0 <= i < 16 and 0 <= j < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j] -> [(i)] }, { S[i, j] -> [(j)] }]\""
                        , "    |  }"
                        , "    |}"
                        ]
                expectedInterchangeThenTile =
                    unlines
                        [ "    |{ domain: \"{ S[i, j] : 0 <= i < 16 and 0 <= j < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j] -> [(j - (j) mod 2)] }, { S[i, j] -> [(i - (i) mod 3)] }]\""
                        , "    |  , child: {"
                        , "    |      schedule: \"[{ S[i, j] -> [((j) mod 2)] }, { S[i, j] -> [((i) mod 3)] }]\""
                        , "    |    }"
                        , "    |  }"
                        , "    |}"
                        ]
                expectedTileThenInterchange =
                    unlines
                        [ "    |{ domain: \"{ S[i, j] : 0 <= i < 16 and 0 <= j < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j] -> [(j - (j) mod 3)] }, { S[i, j] -> [(i - (i) mod 2)] }]\""
                        , "    |  , child: {"
                        , "    |      schedule: \"[{ S[i, j] -> [((j) mod 3)] }, { S[i, j] -> [((i) mod 2)] }]\""
                        , "    |    }"
                        , "    |  }"
                        , "    |}"
                        ]

            result <- runISL $ do
                sched <- schedule $ normalizePrettySchedule inputSchedule
                a <- applyScheduleOptimizations [LoopInterchange [1, 0], Tile [[2], [3]]] sched
                b <- applyScheduleOptimizations [Tile [[2], [3]], LoopInterchange [1, 0]] sched
                expectedA <- schedule $ normalizePrettySchedule expectedInterchangeThenTile
                expectedB <- schedule $ normalizePrettySchedule expectedTileThenInterchange
                eqA <- scheduleIsEqual a expectedA
                eqB <- scheduleIsEqual b expectedB
                pure (eqA, eqB)
            result `shouldBe` Right (True, True)

        it "fails on invalid loop interchange permutation" $ do
            let inputSchedule =
                    unlines
                        [ "    |{ domain: \"{ S[i, j] : 0 <= i < 16 and 0 <= j < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j] -> [(i)] }, { S[i, j] -> [(j)] }]\""
                        , "    |  }"
                        , "    |}"
                        ]

            result <- runISL $ do
                sched <- schedule $ normalizePrettySchedule inputSchedule
                void $ applyScheduleOptimizations [LoopInterchange [1, 1]] sched
            isLeft result `shouldBe` True

        it "fails on ragged multi-level tile sizes" $ do
            let inputSchedule =
                    unlines
                        [ "    |{ domain: \"{ S[i, j] : 0 <= i < 16 and 0 <= j < 16 }\""
                        , "    |, child: {"
                        , "    |    schedule: \"[{ S[i, j] -> [(i)] }, { S[i, j] -> [(j)] }]\""
                        , "    |  }"
                        , "    |}"
                        ]

            result <- runISL $ do
                sched <- schedule $ normalizePrettySchedule inputSchedule
                void $ applyScheduleOptimizations [Tile [[8, 4], [8]]] sched
            isLeft result `shouldBe` True

normalizePrettySchedule :: String -> String
normalizePrettySchedule =
    unlines
        . map stripMargin
        . lines
  where
    stripMargin line =
        case dropWhile (== ' ') line of
            ('|' : rest) -> rest
            other -> other
