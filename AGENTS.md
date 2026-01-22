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

# ISL Haskell Bindings

Integer Set Library (ISL) の Haskell バインディング。
多面体解析（polyhedral analysis）の基盤ライブラリ。

## モジュール構成

- `ISL.Core` - ISL計算モナド（`runISL`）、エラー型
- `ISL.*` - 各ドメインの公開API（Set, Map, Schedule等）
- `ISL.Internal.FFI` - C FFI バインディング
- `ISL.Internal.<Module>/Types.hs` - 型定義
- `ISL.Internal.<Module>/Ops.hs` - 操作実装

## 開発規約

### 新機能追加時

新しい関数・モジュールを追加する際は、必ず対応するテストも同時に追加すること。

1. `ISL.Internal.FFI` に `foreign import` を追加
2. `Internal/<Module>/Ops.hs` にラッパー実装
3. 公開APIは `ISL/<Module>.hs` から再エクスポート
4. **テストを `test/ISL/<Module>Spec.hs` に追加**

### テストの書き方

全一致を検証するアサーションを使うこと：

- `shouldBe` - 値の完全一致
- `shouldMatchList` - リストの要素一致（順序不問）

部分一致の `shouldContain` は原則使わない。

## テスト実行

```bash
cabal test isl-test
```
