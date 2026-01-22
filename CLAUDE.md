## Shotoとは

Shotoは多面体コンパイラ（polyhedral compiler）である。テンソル計算を記述し、GPUカーネル/SIMDコードを生成することを目的とする。
今は、C言語をシンプルに出力している

## パッケージ構成

### shoto

メインのコンパイラパッケージ。

### isl

Integer Set Library (ISL) のHaskellバインディング。多面体解析の数学的基盤を提供する。shotoから独立したパッケージとして分離することで、ISLバインディング単体での再利用を可能にする。

## 言語・スタイル

- **GHC2024**: 最新のHaskell言語拡張セットを使用
- **Leading comma**: カンマ先頭スタイル（差分が見やすい）
- **4スペースインデント**

## 前提

`nix develop`環境内で実行すること。lefthookによりコミット時にフォーマットとテストが自動実行される。

## テスト

```bash
cabal test all
```

## lint

```bash
hlint .
```

自動で修正を試みる
```bash
./hlint-refactor.sh
```

## フォーマット

```bash
./format.sh
```
