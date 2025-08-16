module IR (Node (..), IR, IRBuildState (..), IRBuilder, ValueId) where

import           Control.Monad.State (State)
import qualified Data.Map            as M

type ValueId = Int

-- Operationで[ValueId]の順番によって、入力されたValueがどの入力かを管理する
-- TODO: Constを導入する
data Node i op
    = Input i
    | Operation op [ValueId]

-- そのうちConstを追加する
-- \| Const co

type IR i op = M.Map ValueId (Node i op)

data IRBuildState i op = IRBuildState {nextId :: ValueId, state :: M.Map ValueId (Node i op)}

type IRBuilder i op = State (IRBuildState i op)
