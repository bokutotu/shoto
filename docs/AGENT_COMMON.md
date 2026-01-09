# shoto - Polyhedral Tensor Compiler

## 概要

Haskell/NixによるPolyhedral modelベースのテンソルコンパイラ。
レイアウトとスケジュールの同時最適化を行い、GPU向けコードを自動生成する。

## 技術スタック

- Haskell (GHC 9.12) + Cabal + hpack
- Nix (haskell.nix)
- ISL (Integer Set Library)

## コマンド

| コマンド | 説明 |
|---------|------|
| `nix develop` | 開発環境に入る |
| `cabal build` | ビルド |
| `cabal test` | テスト実行 |
| `hpack` | package.yaml → shoto.cabal 生成 |
| `fourmolu -i src/` | コードフォーマット |

## プロジェクト構造

```
shoto/
├── src/                 # ソースコード
│   ├── Shoto.hs         # メインモジュール
│   ├── Runtime.hs       # ランタイム
│   └── Internal/
│       └── FFI.hs       # FFI (ISL等)
├── test/                # テストコード
└── docs/                # 設計ドキュメント
    ├── OBJECTIVE.md     # プロジェクト目標
    ├── ARCH.md          # アーキテクチャ
    └── TECH_STACK.md    # 技術スタック詳細
```

## 開発ガイドライン

- 実装時は `docs/` を参照する
- 設計変更時は対応するドキュメントを更新する
- コードから読み取れることはコードを読む
- Haskellの標準的なコーディング規約に従う（fourmoluでフォーマット）

## 注意事項

- ISLバインディングはFFI経由で実装
- package.yaml変更後は`hpack`を実行してcabalファイルを再生成
