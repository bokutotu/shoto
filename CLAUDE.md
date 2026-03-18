## 環境

- `nix develop` の中で作業すること。
- `lefthook` は、変更した Haskell ファイルに対して pre-commit で `./format.sh` と `cabal test all` を実行する。

## コマンド

- テスト: `cabal test all`
- lint: `hlint .`
- HLint の自動リファクタ: `./hlint-refactor.sh`
- Haskell のフォーマット: `./format.sh`

## Haskell スタイル

- 新規または編集したコードでは、レコードアクセスにドット構文（`x.field`）を優先する。

## ドキュメント

- `AGENTS.md` と `CLAUDE.md` は意味的に同期させる。
- ルートの AI 向けドキュメントは、ワークフローと規約に絞り、コードから分かる詳細は重複して書かない。
