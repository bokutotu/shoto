module Main (main) where

import Test.HUnit
import Shoto (add)

-- add関数のテストケース: 1 と 2 を足して 3 になるか確認
testAdd :: Test
testAdd = TestCase $ assertEqual "add 1 2 should equal 3" 3 (add 1 2)

-- テストをまとめる（複数テストがある場合はここに追加）
tests :: Test
tests = TestList [ TestLabel "Addition Test" testAdd ]

main :: IO ()
main = do
  testCounts <- runTestTT tests
  if errors testCounts + failures testCounts == 0
    then putStrLn "All tests passed."
    else putStrLn "Some tests failed." >> error "Test suite failed."

