module Shoto (compile) where

import           ISL (runISL, unionMap, unionSet)

compile :: String -> String -> String -> String -> String
compile domain write reed schedule = runISL $ do
    domain <- unionSet domain
    write <- unionMap write
    reed <- unionMap reed
    schedule <- unionMap schedule
