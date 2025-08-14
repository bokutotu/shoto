# 開発計画とステータス

## 開発方針

### インクリメンタル開発アプローチ
9cc的な開発手法で、最も単純な実装から始めて段階的に機能を追加していく。

## 開発フェーズ

### Phase 1: 単一演算の直接コード生成（現在）
**目標**: 1つの演算を含むTiny IRグラフから直接CUDAコードを生成

- Tiny IRのデータ構造実装
- 単一演算（例：要素ごと加算）のグラフ表現
- Tiny IR → CUDAの直接変換（1演算 = 1カーネル）
- 生成コードのコンパイル・実行確認

```
TinyIR (single op) → CUDA kernel
```

### Phase 2: Region分割の導入
**目標**: 複数演算を含むグラフを適切な粒度でRegionに分割

- Regionの概念を実装（複数演算をグループ化）
- 融合可能性の判定ロジック
- 依存関係の解析
- 1 Region = 1 カーネルの原則

```
TinyIR (multi ops) → Region分割 → Region[]
```

### Phase 3: RegionからのCUDA生成
**目標**: Region単位でCUDAコードを生成

- Region内の演算を融合したカーネル生成
- 中間値のレジスタ保持
- Epilogue融合（bias, activation等）

```
Region → CUDA kernel (fused ops)
```

### Phase 4: CUDA方言IRの導入
**目標**: より構造化された中間表現を経由した生成

- CUDA方言IRの設計・実装
  - メモリ階層の明示（gmem/smem/reg）
  - アーキ依存命令の抽象化（cp.async, tma等）
- Region → CUDA方言IR → CUDAコード
- アーキテクチャ（SM80/SM90）による分岐

```
Region → CUDA Dialect IR → CUDA kernel
```

### Phase 5: 最適化の追加
**目標**: 性能最適化の実装

- IndexBookの実装（軸・領域・アクセスの正規化）
- Polyhedral解析
- タイリング、ループ融合
- Schedule Planの導入

## 現在のステータス (2025-01-14)

### 実装済み
- [x] プロジェクト基本構造
- [x] CUDA FFIの基本実装（nvrtc/cudaラッパー）
- [x] Runtime.hsでのコンパイル・実行機能
- [x] TinyIRの基本データ構造定義

### 作業中
- [ ] **Phase 1: 単一演算の直接コード生成**
  - [ ] TinyIRの演算子実装
  - [ ] 単純な要素ごと演算のグラフ構築
  - [ ] CUDA生成関数の実装
  - [ ] 実行テスト

## 直近のタスク

1. **TinyIRで `a + b` を表現**
   - 2つの入力テンソル
   - Binary(Add)演算
   - 1つの出力テンソル

2. **素朴なCUDAカーネル生成**
   ```cuda
   __global__ void add_kernel(float* a, float* b, float* out, int n) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < n) {
       out[idx] = a[idx] + b[idx];
     }
   }
   ```

3. **生成・コンパイル・実行の確認**
   - nvrtcでコンパイル
   - カーネル実行
   - 結果検証

## 設計原則

- **動くものを最速で作る**: 最適化は後回し、まず動作確認
- **1ステップずつ**: 各フェーズが独立して動作確認可能
- **段階的な抽象化**: 必要性が明確になってから抽象化を導入

## 変更履歴

### 2025-01-14
- plan.mdをSPEC.md（仕様書）とplan.md（開発計画）に分離
- Frontend IRをスキップし、Tiny IRから直接開発する方針に変更
- 単一演算→Region→CUDA方言という段階的な開発計画を策定