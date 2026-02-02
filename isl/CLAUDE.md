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

## ISL モナド (`ISL s`)

ISL モナドは ISL ライブラリの `isl_ctx` を内部で管理する。ユーザーは `isl_ctx` を直接操作する必要はない。

- `runISL :: ISL s a -> IO (Either IslError a)` で ISL 計算を実行
- `isl_ctx` の確保・解放は自動的に行われる
- FFI 関数が `isl_ctx` を必要とする場合、`ISL.Core.askEnv` で取得できる

### 注意: context という名前の関数

ISL には複数の「context」という名前を持つ関数がある：

- `isl_ctx_*`: ISL ライブラリ全体のコンテキスト（ISL モナドが管理）
- `isl_schedule_constraints_set_context`: パラメータ制約の Set を設定（`isl_set` 型）
- `isl_ast_build_from_context`: AST 生成時のパラメータ制約（`isl_set` 型）

`isl_schedule_constraints_set_context` や `isl_ast_build_from_context` の引数は `isl_ctx` ではなく `isl_set` である。
