{-# LANGUAGE OverloadedRecordDot #-}

module Graph (Graph, nodeSet, filterGraph, Node (..)) where

import qualified Data.Set as S

data Node a = Node {id :: Int, content :: a} deriving (Show)

instance Eq (Node a) where
    n1 == n2 = n1.id == n2.id

instance Ord (Node a) where
    compare n1 n2 = compare (Graph.id n1) (Graph.id n2)

data Graph a
    = Empty
    | Vertex (Node a)
    | Overlay (Graph a) (Graph a)
    | Connect (Graph a) (Graph a)
    deriving (Eq, Show)

nodeSet :: Graph a -> S.Set (Node a)
nodeSet = inner S.empty
  where
    inner :: S.Set (Node a) -> Graph a -> S.Set (Node a)
    inner s Empty = s
    inner s (Vertex x) = S.insert x s
    inner s (Overlay g1 g2) = S.union (S.union (inner S.empty g1) (inner S.empty g2)) s
    inner s (Connect g1 g2) = S.union (S.union (inner S.empty g1) (inner S.empty g2)) s

filterGraph :: (a -> Bool) -> Graph a -> S.Set (Node a)
filterGraph fn g = S.filter (\node -> fn node.content) (nodeSet g)
