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
- **OverloadedRecordDot**: 新規/編集するHaskellコードでは、レコードフィールドアクセスにドット構文（`x.field`）を優先して使う
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

## FrontendIR テンソル規約

- `Program` は `tensors :: NonEmpty TensorDecl` でテンソル宣言を必須とする。
- テンソル形状はシンボリックな `ParamName` の配列で、ランクは `length shape`。
- 検証境界は `checkProgram`（重複、未宣言テンソル、ランク不一致、未知 shape パラメータ）。
- `lowerToRaw` は検証済みプログラムを lower するだけで、検証は行わない。

## ドキュメント更新

計画（plan mode）が完了するたびに、必要に応じて以下のファイルを更新すること：

- `/CLAUDE.md` - Claude Code 用の指示
- `/AGENTS.md` - 複数 AI ツール共有の指示
- `/isl/CLAUDE.md` - ISL パッケージ固有の規約

`AGENTS.md` と `CLAUDE.md` の共通部分は同期を保つこと。
