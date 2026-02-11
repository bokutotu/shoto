module PolyhedralSpec (spec) where

import           ISL        (AstExpression (..), AstOp (..), AstTree (..),
                             runISL)
import           Polyhedral (RawPolyhedralModel (..), ScheduleOptimization (..),
                             synthesize)
import           Test.Hspec

spec :: Spec
spec = do
    describe "Polyhedral synthesize" $ do
        it "simple 1D loop" $ do
            let raw =
                    RawPolyhedralModel
                        { context = "{ : }"
                        , domain = "[N] -> { S[i] : 0 <= i < N }"
                        , programOrder = "{ S[i] -> [i] }"
                        , readAccess = "{ S[i] -> B[i] }"
                        , writeAccess = "{ S[i] -> A[i] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S") [ExprId "c0"]
                        }

            result <- runISL $ synthesize [] raw
            result `shouldBe` Right expectedAst

        it "simple 2D loop (no axis swap)" $ do
            let raw =
                    RawPolyhedralModel
                        { context = "{ : }"
                        , domain = "[N,M] -> { S[i,j] : 0 <= i < N and 0 <= j < M }"
                        , programOrder = "{ S[i,j] -> [i,j] }"
                        , readAccess = "{ S[i,j] -> B[i,j] }"
                        , writeAccess = "{ S[i,j] -> A[i,j] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

                innerFor =
                    AstFor
                        { forIterator = "c1"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "M"))
                        , forInc = ExprInt 1
                        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S") [ExprId "c0", ExprId "c1"]
                        }
                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = innerFor
                        }

            result <- runISL $ synthesize [] raw
            result `shouldBe` Right expectedAst

        it "simple 2D loop with loop interchange optimization" $ do
            let raw =
                    RawPolyhedralModel
                        { context = "{ : }"
                        , domain = "[N,M] -> { S[i,j] : 0 <= i < N and 0 <= j < M }"
                        , programOrder = "{ S[i,j] -> [i,j] }"
                        , readAccess = "{ S[i,j] -> B[i,j] }"
                        , writeAccess = "{ S[i,j] -> A[i,j] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

                innerFor =
                    AstFor
                        { forIterator = "c1"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S") [ExprId "c1", ExprId "c0"]
                        }
                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "M"))
                        , forInc = ExprInt 1
                        , forBody = innerFor
                        }

            result <- runISL $ synthesize [LoopInterchange [1, 0]] raw
            result `shouldBe` Right expectedAst

        it "simple 2D loop with loop interchange and multi-level tiling optimization" $ do
            let raw =
                    RawPolyhedralModel
                        { context = "{ : }"
                        , domain = "{ S[i,j] : 0 <= i < 64 and 0 <= j < 32 }"
                        , programOrder = "{ S[i,j] -> [i,j] }"
                        , readAccess = "{ S[i,j] -> B[i,j] }"
                        , writeAccess = "{ S[i,j] -> A[i,j] }"
                        , reductionDomain = "{ }"
                        , reductionRead = "{ }"
                        , reductionWrite = "{ }"
                        }

                pointITileFor =
                    AstFor
                        { forIterator = "c5"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLe (ExprId "c5") (ExprInt 1))
                        , forInc = ExprInt 1
                        , forBody =
                            AstUser $
                                ExprOp $
                                    OpCall
                                        (ExprId "S")
                                        [ ExprOp (OpAdd (ExprOp (OpAdd (ExprId "c1") (ExprId "c3"))) (ExprId "c5"))
                                        , ExprOp (OpAdd (ExprOp (OpAdd (ExprId "c0") (ExprId "c2"))) (ExprId "c4"))
                                        ]
                        }
                pointJTileFor =
                    AstFor
                        { forIterator = "c4"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLe (ExprId "c4") (ExprInt 3))
                        , forInc = ExprInt 1
                        , forBody = pointITileFor
                        }
                innerITileFor =
                    AstFor
                        { forIterator = "c3"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLe (ExprId "c3") (ExprInt 7))
                        , forInc = ExprInt 2
                        , forBody = pointJTileFor
                        }
                innerJTileFor =
                    AstFor
                        { forIterator = "c2"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLe (ExprId "c2") (ExprInt 15))
                        , forInc = ExprInt 4
                        , forBody = innerITileFor
                        }
                outerITileFor =
                    AstFor
                        { forIterator = "c1"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLe (ExprId "c1") (ExprInt 63))
                        , forInc = ExprInt 8
                        , forBody = innerJTileFor
                        }
                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLe (ExprId "c0") (ExprInt 31))
                        , forInc = ExprInt 16
                        , forBody = outerITileFor
                        }

            result <- runISL $ synthesize [LoopInterchange [1, 0], Tile [[16, 4], [8, 2]]] raw
            result `shouldBe` Right expectedAst

        it "fused gemm + add + relu (no axis swap)" $ do
            let raw =
                    RawPolyhedralModel
                        { context = "[N,M,K] -> { : 0 < K }"
                        , domain =
                            "[N,M,K] -> { S_init[i,j] : 0 <= i < N and 0 <= j < M; S_gemm[i,j,k] : 0 <= i < N and 0 <= j < M and 0 <= k < K; S_out[i,j] : 0 <= i < N and 0 <= j < M }"
                        , programOrder =
                            "[N,M,K] -> { S_init[i,j] -> [i,j,0]; S_gemm[i,j,k] -> [i,j,k+1]; S_out[i,j] -> [i,j,K+1] }"
                        , readAccess =
                            "[N,M,K] -> { S_gemm[i,j,k] -> A[i,k]; S_gemm[i,j,k] -> B[k,j]; S_gemm[i,j,k] -> C[i,j]; S_out[i,j] -> C[i,j]; S_out[i,j] -> Bias[i,j] }"
                        , writeAccess = "[N,M,K] -> { S_init[i,j] -> C[i,j]; S_gemm[i,j,k] -> C[i,j]; S_out[i,j] -> Y[i,j] }"
                        , reductionDomain = "[N,M,K] -> { S_gemm[i,j,k] }"
                        , reductionRead = "[N,M,K] -> { S_gemm[i,j,k] -> C[i,j] }"
                        , reductionWrite = "[N,M,K] -> { S_gemm[i,j,k] -> C[i,j] }"
                        }

                innerKLoop =
                    AstFor
                        { forIterator = "c2"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c2") (ExprId "K"))
                        , forInc = ExprInt 1
                        , forBody = AstUser $ ExprOp $ OpCall (ExprId "S_gemm") [ExprId "c0", ExprId "c1", ExprId "c2"]
                        }
                innerJFor =
                    AstFor
                        { forIterator = "c1"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "M"))
                        , forInc = ExprInt 1
                        , forBody =
                            AstBlock
                                [ AstUser $ ExprOp $ OpCall (ExprId "S_init") [ExprId "c0", ExprId "c1"]
                                , AstBlock
                                    [ innerKLoop
                                    , AstUser $ ExprOp $ OpCall (ExprId "S_out") [ExprId "c0", ExprId "c1"]
                                    ]
                                ]
                        }
                expectedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = innerJFor
                        }

            result <- runISL $ synthesize [] raw
            result `shouldBe` Right expectedAst
