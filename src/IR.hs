module IR (Node (..), IR, IRBuildState (..), IRBuilder, NodeId (..)) where

import           Control.Monad.State (State)
import qualified Data.Map            as M

newtype NodeId = NodeId {unId :: Int} deriving (Eq, Show, Ord)

-- Operationで[NodeId]の順番によって、入力されたNodeがどの入力かを管理する
-- TODO: Constを導入する
data Node i op
    = Input i
    | Operation op [NodeId]

-- そのうちConstを追加する
-- \| Const co

type IR i op = M.Map NodeId (Node i op)

data IRBuildState i op = IRBuildState {nextId :: NodeId, state :: M.Map NodeId (Node i op)}

type IRBuilder i op = State (IRBuildState i op)
