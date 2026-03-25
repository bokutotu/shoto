{-# LANGUAGE OverloadedStrings #-}

module Codegen.GenIRSpec (spec) where

import           Codegen.GenIR       (GenExpr (..), GenIRError (..),
                                      GenProgram (..), GenStmt (..),
                                      GenTensorDecl (..), GenTensorRef (..),
                                      buildGenProgram)
import qualified Data.List.NonEmpty  as NE
import           FrontendIR          (Axis (..), Expr (..), IxExpr (..),
                                      Program (..), ReductionOp (..), Stmt (..),
                                      TensorDecl (..))
import           Polyhedral.Internal (AstExpression (..), AstOp (..),
                                      AstTree (..))
import           Test.Hspec

spec :: Spec
spec = do
    describe "Codegen.GenIR" $ do
        it "builds a lowered AST from an untiled perfect loop nest" $ do
            buildGenProgram simpleCopy2DAst simpleCopy2DProgram
                `shouldBe` Right
                    GenProgram
                        { genExtentParams = ["N", "M"]
                        , genTensorDecls =
                            [ GenTensorDecl{genTensor = "A", genShape = ["N", "M"]}
                            , GenTensorDecl{genTensor = "B", genShape = ["N", "M"]}
                            ]
                        , genBody =
                            [ GenFor
                                { genIter = "c0"
                                , genBound = "N"
                                , genBody =
                                    [ GenFor
                                        { genIter = "c1"
                                        , genBound = "M"
                                        , genBody =
                                            [ GenAssign
                                                { genTarget =
                                                    GenTensorRef
                                                        { genTensor = "A"
                                                        , genIndices = ["c0", "c1"]
                                                        }
                                                , genExpr =
                                                    GenLoad
                                                        GenTensorRef
                                                            { genTensor = "B"
                                                            , genIndices = ["c0", "c1"]
                                                            }
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }

        it "keeps loop schedule order separate from logical tensor indexing" $ do
            let interchangedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "M"))
                        , forInc = ExprInt 1
                        , forBody =
                            AstFor
                                { forIterator = "c1"
                                , forInit = ExprInt 0
                                , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "N"))
                                , forInc = ExprInt 1
                                , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c1", ExprId "c0"]
                                }
                        }

            buildGenProgram interchangedAst simpleCopy2DProgram
                `shouldBe` Right
                    GenProgram
                        { genExtentParams = ["N", "M"]
                        , genTensorDecls =
                            [ GenTensorDecl{genTensor = "A", genShape = ["N", "M"]}
                            , GenTensorDecl{genTensor = "B", genShape = ["N", "M"]}
                            ]
                        , genBody =
                            [ GenFor
                                { genIter = "c0"
                                , genBound = "M"
                                , genBody =
                                    [ GenFor
                                        { genIter = "c1"
                                        , genBound = "N"
                                        , genBody =
                                            [ GenAssign
                                                { genTarget =
                                                    GenTensorRef
                                                        { genTensor = "A"
                                                        , genIndices = ["c1", "c0"]
                                                        }
                                                , genExpr =
                                                    GenLoad
                                                        GenTensorRef
                                                            { genTensor = "B"
                                                            , genIndices = ["c1", "c0"]
                                                            }
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }

        it "ignores the statement call name while lowering the frontend stmt template" $ do
            let renamedStmtAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody = AstUser $ ExprOp $ OpCall (ExprId "copy_stmt") [ExprId "c0"]
                        }

            buildGenProgram renamedStmtAst simpleCopyProgram
                `shouldBe` Right
                    GenProgram
                        { genExtentParams = ["N"]
                        , genTensorDecls =
                            [ GenTensorDecl{genTensor = "A", genShape = ["N"]}
                            , GenTensorDecl{genTensor = "B", genShape = ["N"]}
                            ]
                        , genBody =
                            [ GenFor
                                { genIter = "c0"
                                , genBound = "N"
                                , genBody =
                                    [ GenAssign
                                        { genTarget =
                                            GenTensorRef
                                                { genTensor = "A"
                                                , genIndices = ["c0"]
                                                }
                                        , genExpr =
                                            GenLoad
                                                GenTensorRef
                                                    { genTensor = "B"
                                                    , genIndices = ["c0"]
                                                    }
                                        }
                                    ]
                                }
                            ]
                        }

        it "lowers reductions into GenReduction nodes" $ do
            buildGenProgram simpleCopy2DAst sumRowsProgram
                `shouldBe` Right
                    GenProgram
                        { genExtentParams = ["N", "M"]
                        , genTensorDecls =
                            [ GenTensorDecl{genTensor = "A", genShape = ["N"]}
                            , GenTensorDecl{genTensor = "B", genShape = ["N", "M"]}
                            ]
                        , genBody =
                            [ GenFor
                                { genIter = "c0"
                                , genBound = "N"
                                , genBody =
                                    [ GenFor
                                        { genIter = "c1"
                                        , genBound = "M"
                                        , genBody =
                                            [ GenReduction
                                                { genReductionOp = ReduceAdd
                                                , genTarget =
                                                    GenTensorRef
                                                        { genTensor = "A"
                                                        , genIndices = ["c0"]
                                                        }
                                                , genExpr =
                                                    GenLoad
                                                        GenTensorRef
                                                            { genTensor = "B"
                                                            , genIndices = ["c0", "c1"]
                                                            }
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }

        it "rejects non-identifier stmt-call arguments" $ do
            let tiledLikeAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody =
                            AstUser $
                                ExprOp $
                                    OpCall
                                        (ExprId "S0")
                                        [ExprOp (OpAdd (ExprId "c0") (ExprInt 1))]
                        }

            buildGenProgram tiledLikeAst simpleCopyProgram
                `shouldBe` Left ErrGenExpectedCallArgumentIdentifier

        it "rejects non-perfect loop nests" $ do
            let malformedAst =
                    AstFor
                        { forIterator = "c0"
                        , forInit = ExprInt 0
                        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                        , forInc = ExprInt 1
                        , forBody =
                            AstIf
                                { ifCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
                                , ifThen = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0"]
                                , ifElse = Nothing
                                }
                        }

            buildGenProgram malformedAst simpleCopyProgram
                `shouldBe` Left ErrGenMalformedLoopNest

simpleCopy2DAst :: AstTree
simpleCopy2DAst =
    AstFor
        { forIterator = "c0"
        , forInit = ExprInt 0
        , forCond = ExprOp (OpLt (ExprId "c0") (ExprId "N"))
        , forInc = ExprInt 1
        , forBody =
            AstFor
                { forIterator = "c1"
                , forInit = ExprInt 0
                , forCond = ExprOp (OpLt (ExprId "c1") (ExprId "M"))
                , forInc = ExprInt 1
                , forBody = AstUser $ ExprOp $ OpCall (ExprId "S0") [ExprId "c0", ExprId "c1"]
                }
        }

simpleCopyProgram :: Program
simpleCopyProgram =
    Program
        { axes =
            NE.fromList
                [Axis{iter = "i", extent = "N"}]
        , tensors =
            NE.fromList
                [ TensorDecl{tensor = "A", shape = ["N"]}
                , TensorDecl{tensor = "B", shape = ["N"]}
                ]
        , stmts =
            NE.fromList
                [ Assign
                    { outputTensor = "A"
                    , outputIndex = [IxVar "i"]
                    , rhs = ELoad "B" [IxVar "i"]
                    }
                ]
        }

simpleCopy2DProgram :: Program
simpleCopy2DProgram =
    Program
        { axes =
            NE.fromList
                [ Axis{iter = "i", extent = "N"}
                , Axis{iter = "j", extent = "M"}
                ]
        , tensors =
            NE.fromList
                [ TensorDecl{tensor = "A", shape = ["N", "M"]}
                , TensorDecl{tensor = "B", shape = ["N", "M"]}
                ]
        , stmts =
            NE.fromList
                [ Assign
                    { outputTensor = "A"
                    , outputIndex = [IxVar "i", IxVar "j"]
                    , rhs = ELoad "B" [IxVar "i", IxVar "j"]
                    }
                ]
        }

sumRowsProgram :: Program
sumRowsProgram =
    Program
        { axes =
            NE.fromList
                [ Axis{iter = "i", extent = "N"}
                , Axis{iter = "j", extent = "M"}
                ]
        , tensors =
            NE.fromList
                [ TensorDecl{tensor = "A", shape = ["N"]}
                , TensorDecl{tensor = "B", shape = ["N", "M"]}
                ]
        , stmts =
            NE.fromList
                [ Reduction
                    { reductionOp = ReduceAdd
                    , outputTensor = "A"
                    , outputIndex = [IxVar "i"]
                    , rhs = ELoad "B" [IxVar "i", IxVar "j"]
                    }
                ]
        }
