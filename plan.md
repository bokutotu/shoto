# 深層学習コンパイラ 仕様書

本仕様は、\*\*最小語彙の Tinygrad‑like IR（Movement / Unary / Binary / Reduce のみ）\*\*を起点に、**IndexBook による軸（添え字）正規化 → Poly‑View（解析専用IR） → Region Buffer SSA → GPU方言IR**の順に段階化し、**単一MMAテンプレート**へ **Schedule Plan** を注入して高性能コード（CUDA/C, 一部 inline PTX）を自動生成するための設計である。
以降、記法・語彙は本仕様内で完結するよう統一する。

---

## 0. スコープと原則

* 対象GPU: **NVIDIA SM80（Ampere）/SM90（Hopper）** 以降
* 主精度: **FP16/BF16**（必要箇所で FP32 蓄積）
* 外部 DNN/BLAS: 不使用（**CUDA/C 自動生成**、必要箇所のみ inline PTX）
* 性能方針: **単一MMAテンプレート**に **Schedule Plan** を注入。分岐は **アーキ（SM80/SM90）** と **tail** のみに限定
* 代表最適化: タイル化、`cp.async`/`ldmatrix`/`mma.sync`（SM80）、`tma.load`/`mbarrier`/`wgmma.mma_async`（SM90）、SMEM 二重/三重バッファ、epilogue 融合
* MVP 対象: **GEMM / Conv / Attention**

### 0.1 用語（本仕様での定義）

* **ハロー（halo）**: 出力タイル計算に必要な、入力側の追加読取り“縁”。例：2D, stride=1, dilation=1 の必要入力は `(Th+Kh−1)×(Tw+Kw−1)`。
* **ステージ（pipeline stages）**: 非同期ロードと演算の重ね段数（2=二重、3=三重）。
* **fan‑out**: あるノードの出力を複数の下流が消費（DAG出次数>1）。
* **SCoP**: 添字・境界が（区分）アフィンで表せる部分。
* **UCC**: 単一消費者鎖（分岐しない下流パス）。

---

## 1. IRスタックと役割（7層 + サービス）

1. **Frontend IR**（高位演算・計算グラフ）
2. **Tinygrad‑like IR（Tiny IR, 最小語彙）**
3. **IndexBook（軸自動生成・正規化：サイドテーブル）**
4. **Poly‑View（解析専用IR：SCoP/アクセス式の区分アフィン表現）**
5. **Region Buffer SSA**（出力のみ MemRef、内部は値SSA）
6. **Poly Core**（多面体解析サービス；合法性判定と軽ヒント）
7. **Schedule Plan**（薄い JSON/DSL；GPU方言IRのパラメタ）
8. **GPU方言IR**（単一MMAテンプレートに Plan 属性を注入）

> 以降、1.1〜1.8 で各層を定義し、2 以降で変換規則・アルゴリズムを詳述する。

---

## 1.1 Frontend IR（高位演算）

### 役割

* Conv / Attention / LayerNorm / Elementwise / Reduce 等の**高位属性**を保持（依存DAG）。
* 後段で **最小語彙 Tiny IR** に正規化可能であること。

### 型・主な Op（要点）

* `TensorType{dtype, shape, layout?}`
* `Conv2D(x,w,bias?, attrs{stride,pad,dilation,groups,layout,algo_set?})`
* `Attention(Q,K,V, mask?, attrs{heads, causal?, dropout?, algo_set?})`
* `LayerNorm(x, gamma, beta, attrs{axis, eps})`
* `Elementwise(fn, tensors...)` / `Reduce(fn, x, axes)` / `Movement(kind, ...)`

**不変**: Frontend では実メモリコピーを行わない（Movement は論理操作）。

---

## 1.2 Tinygrad‑like IR（最小語彙）

### 役割

* **Movement / Unary / Binary / Reduce のみ**で構成される純関数DAG（コピー禁止）。
* 行列積・畳み込みなどの縮約は **`reshape/permute/expand/pad` + `mul` + `reduce(sum)`** の**合成パターン**で表現する。

### 使用する Op

* `Movement{kind ∈ {reshape, permute, expand, pad, shrink, view}}`
* `Unary{fn ∈ {relu, gelu, exp, neg, sqrt, recip, …}}`
* `Binary{fn ∈ {add, sub, mul, div, max, where, …}}`
* `Reduce{fn ∈ {sum, max, min}, axes:[…]}`

### 例（GEMM + bias + ReLU）

```json
{
  "ops": [
    {"op":"Movement","name":"a_reshape","kind":"reshape","inputs":["A"],"outputs":["A_r"],"attrs":{"new_shape":["M","K","1"]}},
    {"op":"Movement","name":"b_reshape","kind":"reshape","inputs":["B"],"outputs":["B_r"],"attrs":{"new_shape":["1","K","N"]}},
    {"op":"Binary","name":"mul","fn":"mul","inputs":["A_r","B_r"],"outputs":["T"]},
    {"op":"Reduce","name":"sum_k","fn":"sum","inputs":["T"],"outputs":["C0"],"axes":["K"]},
    {"op":"Movement","name":"bias_expand","kind":"expand","inputs":["bias"],"outputs":["bias_mn"],"attrs":{"new_shape":["M","N"]}},
    {"op":"Binary","name":"bias_add","fn":"add","inputs":["C0","bias_mn"],"outputs":["C1"]},
    {"op":"Unary","name":"relu","fn":"relu","inputs":["C1"],"outputs":["C2"]}
  ],
  "values": {
    "A":{"dtype":"fp16","shape":["M","K"]},
    "B":{"dtype":"fp16","shape":["K","N"]},
    "bias":{"dtype":"fp16","shape":["N"]},
    "A_r":{"dtype":"fp16","shape":["M","K","1"]},
    "B_r":{"dtype":"fp16","shape":["1","K","N"]},
    "T":{"dtype":"fp16","shape":["M","K","N"]},
    "C2":{"dtype":"fp16","shape":["M","N"]}
  }
}
```

---

## 1.3 IndexBook（軸自動生成・正規化：サイドテーブル）

### 目的

* Tiny IR を汚さずに、**軸（添え字）空間を構築時に自動生成・正規化**し、後段の Poly‑View/isl 変換を機械化・安定化する。

### データモデル

* **AxisVar**: `{ id: int, name: "i0|i1|…", size: SymOrInt, kind: "iter"|"reduce"|"broadcast" }`
* **AxisExpr**: アフィン + 整数除算（`floordiv`）のみの式
* **AxisMap**: `AxisExpr[]`（**出力軸 → 入力軸**の式ベクトル）
* **Domain**: `{ constraints: Constr[], pieces?: Domain[] }`（区分アフィン制約集合）
* **IndexInfo（値ごと）**:

  * `axes: AxisVar[]`（出力軸を `i0,i1,...` で連番）
  * `domain: Domain`（`0 ≤ ik < sizek` + 追加ガード）
  * `inputs: { value_id: string, map: AxisMap }[]`（入力ごとのアクセス式）
  * `reduce_axes?: AxisId[]`（Reduce 直前に消える軸の元ID）
* **IndexBook**: `value_id → IndexInfo` の辞書（IR本体と独立に保持）

### 不変条件

1. すべての AxisMap は **アフィン＋floordiv** のみ（`%` は div+ガードに展開）
2. 出力軸の命名は **`i0,i1,...` 連番**。同名は同一空間の同一位置
3. ブロードキャストは **定数0** 成分で表現（式側に条件分岐を持たない）
4. Reduce は対象軸を**削除**し、削除軸IDを `reduce_axes` に記録
5. 非アフィン（gather/scatter/topk 等）は **Non‑SCoP** マーク

### ノード別更新規則

* **Leaf**: 軸 `i0..i{r-1}` を新規発番。`domain: 0 ≤ ik < Dk`、`kind=iter`
* **Unary**: 恒等（軸・Domain・AxisMap 継承）
* **Binary**: 右端揃えブロードキャスト

  * 両辺 size>1 かつ一致 → 新軸 `yt`、両入力は恒等
  * 片側が1 → 新軸 `yt`、放送側は **定数0**、もう一方は恒等
  * 不一致は構築時エラー
* **Movement**

  * `permute`: 置換行列で AxisMap 構成
  * `reshape/view`: **線形化** `L=Σ ik*stridek` → **再分解** `oj=floor(L/stride'_j) % size'_j`（`%` は div+ガードへ展開）
  * `expand`: 新軸 `oj` を導入、AxisMap は **定数0**、Domain に `0 ≤ oj < new_size`
  * `slice/shrink`: 対象軸に `lo ≤ ik < hi` を追加
  * `pad`: 中央は恒等、外側は別ピース（in‑bounds ガード）
* **Reduce**: `axes` を `kind=reduce` にマークし出力から削除。残軸を `i0..` で再番号。`reduce_axes` に記録

---

## 1.4 Poly‑View（解析専用IR）

### 役割

* IndexBook を用いて、**SCoP とアクセス式（区分アフィン）だけ**を保持する解析専用IR。
* `mul → reduce(sum)` の**二項縮約パターン**を検出し、**ContractionPattern**（`pattern ∈ {matmul, conv}`）として**Poly‑View 内だけ**で表す。

### データモデル

* `PolyBlock{name, kind, domain, accesses[], attrs}`

  * `kind ∈ {ewise, reduce, movement_affine, contraction_pattern}`
  * `domain`: 区分アフィン集合
  * `accesses`: 各テンソルへの区分アフィン写像
  * `attrs`（contraction の場合）:
    `{"lhs_idx":[…], "rhs_idx":[…], "out_idx":[…], "reduce_idx":[…], "pattern":"matmul|conv", "affine_offsets":…}`
* `Edge{from, to, deps:{true[], anti[], output[]}}`

**不変**: 非アフィンは SCoP から除外（自然なリージョン境界）。

---

## 1.5 Region Buffer SSA（出力のみ実体化）

### 役割

* **Region ≒ 1カーネル**。**出力のみ MemRef**、内部は**値SSA**。`mul+reduce + epilogue` は同一 Region にまとめる。

### データモデル

* `MemRef{shape, strides, alignment, noalias}`
* `Region{name, inputs:(MemRef|Value)[], outputs:MemRef[], body:[Stmt]}`
* `Stmt = Let(value=Op(...)) | Yield(outputs)`
* 出力に `materialize: "deferred" | "gmem"`（直後消費なら `deferred`）

---

## 1.6 Poly Core（多面体解析サービス）

### 解析項目

* 依存（True/Anti/Output）
* 並列可能軸／縮約軸
* アフィン閉性（Index 合成後も区分アフィン）
* tail 軸抽出
* バリア挿入ヒント（`commit/wait`, `mbarrier`）
* バンク衝突ヒント
* **compute‑at 合法性**と**ハロー量（bytes, per\_axis）**
* **最小バッファ種別**（`reg / smem_ring / gmem`）

### API

```ts
type Axis = string;
type ComputeAt = { ok: boolean, halo: { bytes: number, per_axis: Record<string,number> } };
type MinBuffer = { buffer: "reg"|"smem_ring"|"gmem", depth?: 2|3, bytes?: number };

analyze(region_candidate) -> { ... }
can_compute_at(producer, consumer, tile?) -> ComputeAt
min_buffer_for_edge(producer, consumer, tile?) -> MinBuffer
make_fusable_groups(scop) -> string[][]
```

---

## 1.7 Schedule Plan（薄い JSON/DSL）

### フィールド

* `tile=[BM,BN,BK]`, `stages ∈ {2,3}`
* `bind`（例：`m.o→block.y, n.o→block.x, m.i.o→warp.y, n.i.o→warp.x`）
* `warp_tile`（`64x64` | `64x32`）
* `cache`（例：`{"tensor":"A","where":"smem","at":"k.i","pingpong":true}`）
* `vectorize`（例：`{"axis":"n.i.i","width":8}`）
* `predicate_tail`（例：`["m.i.i","n.i.i","k.i.i"]`）
* `epilogue`（例：`["bias","relu"|"silu"|"residual"|...]`）
* `arch`：`"sm80"|"sm90"`
* `layout_hints`（例：`{"A_swizzle":true,"B_swizzle":true,"C_stride":"row"}`）
* `algo_choice`（**Poly‑View の pattern に対応**）

  * 例：`{"matmul":"implicit_gemm|splitk_*", "conv":"implicit_gemm|winograd_3x3|fft_conv", "attention":"streaming_softmax_2pass|online_softmax"}`
* `local_edges`: `[{from,to,buffer:"reg"|"smem_ring"|"gmem", depth?:2|3}]`

### DSL（最小）

```
split m 128; split n 64; split k 64;
reorder m.o n.o k.o m.i.o n.i.o k.i.o m.i.i n.i.i k.i.i;
bind m.o block.y; bind n.o block.x; bind m.i.o warp.y; bind n.i.o warp.x;
pipeline k.i stages=2;
cache_read A smem at=k.i pingpong=true;
cache_read B smem at=k.i pingpong=true;
vectorize n.i.i 8;
predicate_tail m.i.i n.i.i k.i.i;
epilogue bias relu;
```

**Plan→GPU方言IR 生成規則（要点）**

* `bind` で CTA/warp/lane 構造を決定。
* `pipeline` と `cache_read` で `cp.async|tma.load`、`commit/wait|mbarrier` を挿入。
* `warp_tile` で `ldmatrix|wgmma` 形状を確定。
* `vectorize` で `ld/st.global.v{2,4,8,16}` を選択。
* `predicate_tail` で predicated な末端処理を生成。
* `local_edges` に従い **reg 前渡し / SMEM リング**を組み込む。
* `epilogue` は acc（Cフラグメント）に直接適用（GMEM往復を禁止）。

---

## 1.8 GPU方言IR（単一MMAテンプレート）

### 固定骨格

`cp.async|tma.load → smem ping‑pong(commit/wait | mbarrier) → ldmatrix → {mma.sync | wgmma} → epilogue → st.global.vecN`

### ステートメント

* `CpAsync(dst_smem, src_gmem, bytes, group_id)`（SM80）
* `TmaLoad(dst_smem, desc, coords, mbarrier)`（SM90）
* `CommitGroup/WaitGroup`（SM80）
* `LdMatrix(dst_frag, src_smem, layout)`
* `MmaSync`（SM80） / `Wgmma`（SM90, warpgroup=128）
* `Epilogue(acc, ops=[...])`
* `StGlobalVec(ptr, regs|vec, pred?)`

**不変**: アーキ分岐は本層のみ。tail は `predicate_tail` に従う。

---

## 2. Frontend → Tiny IR 正規化（最小語彙での表現）

* **GEMM**: `reshape/expand + mul + reduce(sum over K)`
* **Conv2D**: `pad + movement(affine, im2col相当) + mul + reduce(sum over IC,KH,KW)`（im2col の実体化は不可）
* **Attention**: `QKᵀ` を上記と同型の `mul+reduce`、softmax は `reduce(max/sum) + unary/binary`
* **LayerNorm**: `reduce（welford or 2‑pass tree） + elementwise`
* **Movement 全般**: reshape/permute/expand/pad/slice を IndexBook 規則（線形化→再分解、div+ガード）で表現

---

## 3. Tiny → Poly‑View / isl 変換（添え字変換アルゴリズム）

### 3.1 Space 正規化

* タプル名と次元役割（set/in/out）を規約化。
* **パラメータ**は `align_params`（名前一致で整列）。
* 本体次元は IndexBook の `i0,i1,...` をそのまま用い、**追加の rename/pullback を不要化**。

### 3.2 式の lowering 規則

* 線形項・定数倍：アフィンの和。
* **floordiv**：`t=floor(E/q)`（q>0）を div として導入。
* **剰余**：`E%q = E − q*floor(E/q)` とし、**ガード** `0 ≤ E − q*floor(E/q) < q` を Domain に追加。
* **reshape/view**：旧次元の線形化 `L=Σ ik*stridek` → 新次元へ再分解 `oj=floor(L/stride'_j) % size'_j`。
* **permute/expand/slice/pad**：IndexBook の AxisMap/Domain を `multi_aff/pw_multi_aff`・`set/union_set` へ機械的に落とす。
* Movement 連鎖は **逐次合成**し、式の肥大化を抑制。

### 3.3 piecewise の扱い

* `pad/slice`、境界 `%` 由来条件はピース分割（`pw_*`/`union_*`）。
* 二項演算前に簡約（`gist/coalesce/remove_redundancies`）を実行し、爆発を抑制。

### 3.4 生成物

* **Domain**：`IndexInfo.domain → isl_set/isl_union_set`
* **Access**：各入力 `AxisMap → isl_(pw_)multi_aff`
* **ContractionPattern 検出**（3.5）

### 3.5 ContractionPattern（mul→reduce）検出

* `Binary(mul)` の直上 `Reduce(sum, axes=K)` を候補化。
* 二入力が Movement 以外を含まないことを確認。
* 二入力 AxisMap の比較で \*\*共通（iter）\*\*と \*\*縮約（reduce）\*\*を同定：

  * 例（GEMM）: `lhs:(m,k,0)`, `rhs:(0,k,n)` → out:`(m,n)`、reduce:`k`
* Conv は `affine_offsets`（`h+kh`, `w+kw` 等）を認識し `pattern="conv"`。
* 検出結果は **Poly‑View 内のみ**（Tiny IR は最小語彙のまま）。

---

## 4. Region 生成アルゴリズム

**目的**: Tiny IR を**最適/準最適粒度**でリージョン化し、**1リージョン=1カーネル**、中間の物質化（GMEM ラウンドトリップ）を最小化。

### 4.1 手順

1. **SCoP 抽出**（IndexBook→Poly‑View）：`/`・`%` は div+ガード化。
2. **縮約パターンの認識**（3.5）。
3. **合法性＆余分読み**：`can_compute_at(p→c)` で合法性と **ハロー量（bytes, per\_axis）**。
4. **差分コスト評価**

   * `benefit = (bytes_write_back + bytes_read_back)/effective_BW`
   * `cost = Δocc_time + layout_penalty + fanout_penalty`
   * `Δtime = cost − benefit` が負なら融合。
5. **選択法**

   * 小窓（10–30ノード）: **ILP/最大閉包**で厳密
   * 全体: **出力起点の後ろ向き貪欲** + **直前1手ロールバック**
   * **STOP 条件**: ①UCCでない ②`can_compute_at` 不許可/ハロー過大 ③`Δtime ≥ 0`
6. **Region Buffer SSA 生成**：出力のみ MemRef、必要に応じ `materialize:"deferred"`。Plan に `local_edges` を付与。
7. **Plan → GPU 方言IR**：`algo_choice` は Poly‑View の `pattern` に基づく。

### 4.2 しきい値（目安）

* `smem_per_CTA` が上限の **\~80%** 超見込み → STOP
* `regs_per_thr` 増で **CTA/SM が 1 に低下** → STOP
* `vectorize.width` が **16→8/4** へ縮小 → STOP
* **SM90**: TMA/WGMMA 要求タイルから外れる → STOP

---

## 5. GPU最適化戦略（SM80/SM90）

* **資源**:
  `smem_per_CTA = (BM*BK + BK*BN) * sizeof(T) * stages (+ epilogue_smem?)`
  `regs_per_thr ≈ base + acc_regs(BM,BN,warp_tile) + epilogue_regs`
  `cta_per_SM = min(SMEM_lim/smem_per_CTA, REG_lim/(regs_thr*thr_per_CTA), BLK_lim)`
  `occ = min(1.0, cta_per_SM * warps_per_CTA / max_warps_SM)`
* **パイプライン**: `pipeline_eff ≈ 1 − 1/stages`、prefetch距離は1–2
* **断片ロード**: SM80 `ldmatrix+mma.sync`（スウィズル必須）／SM90 `tma.load+mbarrier→wgmma`
* **ベクトル化**: `width ∈ {4,8,16}`。Cストアは可能な限り `st.global.vN`
* **バンク衝突**: テンプレ側でスウィズル固定
* **split‑K**: serial / parallel（parallel は `atomicAdd`、蓄積 FP32 推奨）
* **持続カーネル**: 多タイル/小問題で起動回数削減

---

## 6. コストモデルとチューニング

* 基本式:
  `time_est = max( FLOPs/(peak_TF*occ*pipeline_eff),  Bytes/(peak_BW*occ) )`
* 融合の差分評価:
  `Δtime = max(ΔFLOPs/(peak_TF*occ*pipeline_eff), ΔBytes/(peak_BW*occ)) + Δocc_time + penalties − saved_bytes/peak_BW`
  `penalties = fanout_penalty + layout_penalty(TMA/WGMMA, vectorize)`
* 探索空間（MVP）: `tile(3–4) × stages{2,3} × warp_tile{2} × vec{4,8,16} × algo_choice（少数）` → **10〜20点**実測
* 実測: Warmup→反復→中央値。**ShapeSets**（形状クラスタ）で結果をキャッシュ

---

## 7. 具体例

### 7.1 GEMM + bias + ReLU

* Tiny IR: §1.2 例
* Poly‑View: `pattern="matmul"`、ハロー=0
* Region: 1個（`materialize:"deferred"`）
* Plan（SM80 例）: `tile=[128,64,64], stages=2, warp_tile=64x64, vec=8`
  `local_edges=[{"from":"sum_k","to":"relu","buffer":"reg"}]`

### 7.2 Conv 3×3 + SiLU

* Tiny IR: `pad + movement(affine) + mul + reduce(sum over ic,kh,kw) + silu`
* Poly‑View: `pattern="conv"`, ハロー(±1,±1)
* Plan: `algo_choice.conv = implicit_gemm | winograd_3x3`、`local_edges=smem_ring`

### 7.3 Attention（QKᵀ→softmax→V, causal）

* Tiny IR: `mul+reduce(sum over d)`（QKᵀ）、行 softmax（max/sum 2パス）、`mul+reduce`（S\*V）
* Poly‑View: 前半/後半を `pattern="matmul"`、中間は ewise+reduce
* Plan: `algo_choice.attention = streaming_softmax_2pass`

---

## 8. ランタイム／ABI／ShapeSets

* **ConstPool**: weights の連続配置とアライン
* **Memory Arena**: 生存区間解析に基づく再利用
* **ShapeSets**: 動的形状をクラスタ化し Plan/実測をキャッシュ
* **API**: `model_init`, `model_run`, `model_free`（Param‑Block 受け渡し）
* **CUDA Graph**: 形状クラスタごとに 1 個キャプチャして再利用
* **RNG（学習）**: counter‑based（Philox系）で副作用を制御
* **エラー**: SMEM/REG 超過・非アフィン・アライン不整合は明示的診断を返す

---

## 9. デバッグ/検証

* ダンプ: `--dump=frontend,tiny,indexbook,poly_view,region,plan,gpu,cu`
* CPU プレイバック（Region 内算術の JIT/解釈）
* 数値検証: FP32 参照との `rtol=1e-3, atol=1e-3` 比較
* 測定ログ: `FLOPs/Bytes/occ/estimate/actual/plan` を CSV 追記

---

## 10. 落とし穴と対策

* tail 分岐増 → `predicate_tail` 自動生成。中央は `vecN/mma` を維持
* レジスタスピル → `warp_tile` 縮小、acc 分割、epilogue 合成順見直し
* SMEM バンク衝突 → テンプレ側スウィズル固定
* coalescing 崩壊 → `bind` を連続軸に、`vectorize` 幅調整
* split‑K 原子衝突 → 出力タイル分割、ブロック内集約→原子
* 数値安定 → softmax は max‑shift、LN は welford、acc は FP32
* ハロー過大 → `can_compute_at.halo.bytes` 閾値で STOP
* 占有率低下 → `Δocc` で STOP、`stages/warp_tile/vec` を縮退

---

## 11. 実装ロードマップ

1. **IndexBook** の最小実装（§1.3 規則）
2. Tiny → Poly‑View 変換（§3）
3. ContractionPattern 検出（§3.5）
4. `can_compute_at` と **ハロー算出**（conv の pad を含む）
5. Region 生成（§4）と `local_edges` 生成
6. Plan（§1.7）→ GPU方言IR（§1.8）のコード生成
7. SM80 GEMM → epilogue 融合 → 小チューナ（10〜20点）
8. Conv（implicit → winograd\_3x3）
9. Attention 2パス
10. SM90 分岐（`tma.load/mbarrier/wgmma`）
11. CUDA Graph / ShapeSets / Arena / ConstPool

---

## 12. 付録：形式仕様

### 12.1 Tiny IR（最小語彙）

```json
{
  "ops": [
    {"op":"Movement","name":"string","kind":"reshape|permute|expand|pad|shrink|view","inputs":["..."],"outputs":["..."],"attrs":{...}},
    {"op":"Unary","name":"string","fn":"relu|gelu|exp|neg|sqrt|recip|...","inputs":["..."],"outputs":["..."]},
    {"op":"Binary","name":"string","fn":"add|sub|mul|div|max|where|...","inputs":["..."],"outputs":["..."]},
    {"op":"Reduce","name":"string","fn":"sum|max|min","inputs":["..."],"outputs":["..."],"axes":["..."]}
  ],
  "values": {"name":{"dtype":"fp16|bf16|fp32","shape":["..."]}}
}
```

### 12.2 IndexBook（サイドテーブル）

```ts
type AxisVar  = { id: number, name: string, size: SymOrInt, kind: "iter"|"reduce"|"broadcast" };
type AxisExpr = Affine | FloorDiv;       // 剰余は div+ガードで表現
type AxisMap  = AxisExpr[];              // 出力軸 -> 入力軸
type Domain   = { constraints: Constr[], pieces?: Domain[] };
type IndexInfo = {
  axes: AxisVar[],
  domain: Domain,
  inputs: { value_id: string, map: AxisMap }[],
  reduce_axes?: number[]
};
interface IndexBook { get(value_id: string): IndexInfo }
```

### 12.3 Poly‑View

```json
{
  "poly_view": {
    "blocks": [
      {
        "name":"string",
        "kind":"ewise|reduce|movement_affine|contraction_pattern",
        "domain":{"axis":["lower","upper"]},
        "accesses":[{"tensor":"string","map":"piecewise_affine"}],
        "attrs":{"pattern":"matmul|conv","lhs_idx":[],"rhs_idx":[],"out_idx":[],"reduce_idx":[],"affine_offsets":{}}
      }
    ],
    "edges":[{"from":"string","to":"string","deps":{"true":[],"anti":[],"output":[]}}]
  }
}
```

### 12.4 Region Buffer SSA

```json
{
  "region": {
    "name":"string",
    "inputs":[{"name":"A","memref":{"shape":[],"strides":[],"alignment":128,"noalias":true}}],
    "outputs":[{"name":"C","memref":{"shape":[],"strides":[],"alignment":128,"noalias":true},"materialize":"deferred|gmem"}],
    "body":[{"let":"v","op":{/* Tiny準拠 */}}, {"yield":["v -> C"]}]
  }
}
```

### 12.5 Schedule Plan（JSON）

```json
{
  "tile":[BM,BN,BK],
  "stages":2,
  "bind":{"m.o":"block.y","n.o":"block.x","m.i.o":"warp.y","n.i.o":"warp.x"},
  "warp_tile":"64x64",
  "cache":[{"tensor":"A","where":"smem","at":"k.i","pingpong":true},{"tensor":"B","where":"smem","at":"k.i","pingpong":true}],
  "vectorize":{"axis":"n.i.i","width":8},
  "predicate_tail":["m.i.i","n.i.i","k.i.i"],
  "epilogue":["bias","relu"],
  "arch":"sm80|sm90",
  "layout_hints":{"A_swizzle":true,"B_swizzle":true,"C_stride":"row"},
  "algo_choice":{"matmul":"implicit_gemm|splitk_*","conv":"implicit_gemm|winograd_3x3|fft_conv","attention":"streaming_softmax_2pass|online_softmax"},
  "local_edges":[{"from":"sum_k","to":"relu","buffer":"reg"}]
}
```

### 12.6 DSL 最小文法

```
program := stmt (';' stmt)*
stmt    := split | reorder | fuse | bind | pipeline | cache_read | vectorize | unroll | predicate_tail | epilogue | algo_choice
split   := 'split' axis int
reorder := 'reorder' axis+
fuse    := 'fuse' axis axis '->' axis
bind    := 'bind' axis target
pipeline:= 'pipeline' axis 'stages=' int
cache_read := 'cache_read' tensor 'smem' 'at=' axis ('pingpong=' ('true'|'false'))?
vectorize:= 'vectorize' axis int
unroll  := 'unroll' axis int
predicate_tail := 'predicate_tail' axis+
epilogue:= 'epilogue' op+
algo_choice := 'algo_choice' key value
axis    := [a-z.]+
target  := 'block.x'|'block.y'|'block.z'|'warp.x'|'warp.y'|'warp.z'
op      := 'bias'|'relu'|'silu'|'gelu'|'residual'
key     := 'matmul'|'conv'|'attention'
value   := [a-z0-9_]+
```

### 12.7 isl API 対応表（参考）

| 機能              | 代表 API                                                       |
| --------------- | ------------------------------------------------------------ |
| Domain 作成       | `isl_set_read_from_str` / builder                            |
| Param 整列        | `isl_set_align_params` / `isl_map_align_params`              |
| アフィン・div        | `isl_aff`, `isl_pw_aff`, `isl_multi_aff`, `isl_pw_multi_aff` |
| 合成/引き戻し         | `*_pullback_*`, `isl_set_preimage_multi_aff`                 |
| piecewise/union | `isl_union_set`, `isl_union_map`                             |
| 簡約              | `isl_*_coalesce`, `isl_*_gist`, `isl_*_remove_redundancies`  |

---

## 13. テストと失敗条件（Index/isl 関連）

* **ブロードキャスト不一致**（両辺 size>1 かつ非一致）→ Tiny IR 構築時に即エラー
* **負のストライド**（未対応）→ Movement 段で拒否
* **div ガード欠落**（`%` 展開漏れ）→ Poly‑View 生成時の検証で停止
* **piecewise 爆発**（pad/slice 多用）→ `gist/coalesce/remove_redundancies` を必須実行
* **非SCoP**（データ依存 index）→ 当該点で Region 境界確定

