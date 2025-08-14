# 深層学習コンパイラ 仕様書（全面改訂版）

## 0. スコープと原則（不変）

* 対象 GPU: **NVIDIA SM80（Ampere）/ SM90（Hopper） 以降**
* 主精度: **FP16 / BF16**（必要箇所で **FP32 蓄積**）
* 外部ライブラリ: 不使用（**CUDA/C 自動生成**、必要箇所のみ inline PTX）
* 性能方針: **単一MMAテンプレート**に **Schedule Plan** を注入。分岐は **アーキ（SM80/SM90）** と **tail** のみに限定
* MVP: **GEMM / Conv / Attention**（Epilogue 融合を原則）

### 0.1 用語（本仕様の定義）

* **ハロー（halo）**: 出力タイル計算に必要な入力側の追加読取り“縁”。
* **ステージ（pipeline stages）**: 非同期ロードと演算の重ね段数（2=二重、3=三重）。
* **fan‑out**: ある値の出力を複数の下流が消費（DAG出次数>1）。
* **SCoP**: 添字・境界が（区分）アフィンで表せる部分。
* **UCC**: 単一消費者鎖（分岐しない下流パス）。

---

# 1. アーキ概要と責務分離

本コンパイラは「**意味（What）**」「**添え字と領域（Where）**」「**実装（How）**」を厳密に分離する。

1. **Frontend IR** … モデルの意味（高位演算）。\*\*グラフ署名（I/O）\*\*を持つ。
2. **Tiny IR** … **Movement / Unary / Binary / Reduce**の最小語彙で純関数DAG化。
3. **IndexBook（本仕様の中核）** … 各値の**軸（Axis）・領域（Domain）・アクセス式（AxisMap）**をサイドテーブルで**正規化**。
4. **Poly‑View** … **SCoP とアクセス式のみ**を保持。`mul→reduce(sum)` を **ContractionPattern**に抽象化。
5. **Region Buffer SSA** … **1 Region = 1 カーネル**。出力のみ MemRef、内部は値SSA。
6. **Poly Core** … 依存解析／ハロー算出／最小バッファ分類／バリア・バンク衝突ヒント。
7. **Schedule Plan（JSON/DSL）** … タイル・割付・パイプライン等の方針。
8. **GPU方言IR** … 固定骨格（`cp.async|tma → smem ping‑pong → ldmatrix|wgmma → epilogue → st.global`）に Plan を注入。

**設計原則**

* **非アフィン**は Poly‑View から除外し、そこで **Region 境界**を確定。
* \*\*物質化（gmem ラウンドトリップ）\*\*は最小化。Epilogue は **acc 直接適用**。
* **Front→Tiny→IndexBook**の流れは\*\*決定的（deterministic）\*\*で再現可能。

---

# 2. Frontend IR（高位演算）

## 2.1 役割

* Conv / Attention / LayerNorm / Elementwise / Reduce 等の**高位属性**（例：stride, pad, heads, axis, eps）を保持。
* **実メモリコピー**は行わない（Movement は“論理ビュー”のみ）。

## 2.2 グラフ署名（I/O）— 必須

* **MUST**: **Graph Signature** として、モデルの **入力/出力**を明示する。
* 署名は ABI・ShapeSets・最適化境界の**基準**。
* `Input/Output` を**演算ノード化しない**（デバッグ用としての仮ノードは **MAY**）。

### 署名＋テンソル表＋DAG（例：GEMM+Bias+ReLU）

```json
{
  "signature": {
    "inputs": [
      {"tensor":"A","role":"data","mutability":"immutable"},
      {"tensor":"B","role":"data","mutability":"immutable"},
      {"tensor":"bias","role":"param","mutability":"immutable","storage":"const_pool"}
    ],
    "outputs": [{"tensor":"C2"}]
  },
  "tensors": {
    "A":{"dtype":"fp16","shape":["M","K"]},
    "B":{"dtype":"fp16","shape":["K","N"]},
    "bias":{"dtype":"fp16","shape":["N"]},
    "C2":{"dtype":"fp16","shape":["M","N"]}
  },
  "graph": [
    {"op":"GEMM","name":"gemm","inputs":["A","B"],"outputs":["C0"],"attrs":{"acc_dtype":"fp32"}},
    {"op":"Elementwise","name":"bias_add","fn":"add","inputs":["C0","bias"],"outputs":["C1"]},
    {"op":"Elementwise","name":"relu","fn":"relu","inputs":["C1"],"outputs":["C2"]}
  ]
}
```

## 2.3 不変条件

* すべてのノードは**純関数**。副作用なし。
* 形状は整数と記号（Sym）で表す。未確定は **ShapeSets** でクラスタ化。
* Epilogue 候補は**融合可能**な位置に置く。

---

# 3. Tiny IR（最小語彙）

## 3.1 役割

* **Movement / Unary / Binary / Reduce**のみで、演算を純関数DAGとして記述。
* GEMM/Conv/Attention の縮約は `movement(affine系) + mul + reduce(sum)` の**合成**で表現。

## 3.2 セマンティクス

* **Movement**: `reshape/permute/expand/pad/slice`。**論理ビューのみ**。
* **Unary/Binary**: 要素関数
* **Reduce**: 指定軸の縮約（`sum/max/min`）。縮約軸は**出力から削除**。

## 3.3 エラー条件

* 負ストライド `view`（未対応）
* `slice` の `step>1` が **floordiv+ガード**へ正規化不能

### 参考ダンプ（Attention核）

```json
{
  "ops": [
    {"op":"Movement","name":"expand_q","kind":"expand","inputs":["Q"],"outputs":["Qe"],"attrs":{"new_shape":["B","H","M","N","D"]}},
    {"op":"Movement","name":"expand_k","kind":"expand","inputs":["K"],"outputs":["Ke"],"attrs":{"new_shape":["B","H","M","N","D"]}},
    {"op":"Binary","name":"mul_qk","fn":"mul","inputs":["Qe","Ke"],"outputs":["E"]},
    {"op":"Reduce","name":"sum_d","fn":"sum","inputs":["E"],"outputs":["S"],"axes":["D"]},
    {"op":"Reduce","name":"row_max","fn":"max","inputs":["S"],"outputs":["Mrow"],"axes":["N"]},
    {"op":"Binary","name":"shift","fn":"sub","inputs":["S","Mrow"],"outputs":["S0"]},
    {"op":"Unary","name":"exp","fn":"exp","inputs":["S0"],"outputs":["A"]},
    {"op":"Reduce","name":"row_sum","fn":"sum","inputs":["A"],"outputs":["Z"],"axes":["N"]},
    {"op":"Binary","name":"prob","fn":"div","inputs":["A","Z"],"outputs":["P"]},
    {"op":"Movement","name":"expand_v","kind":"expand","inputs":["V"],"outputs":["Ve"],"attrs":{"new_shape":["B","H","M","N","D"]}},
    {"op":"Binary","name":"pv","fn":"mul","inputs":["P","Ve"],"outputs":["PV"]},
    {"op":"Reduce","name":"sum_n","fn":"sum","inputs":["PV"],"outputs":["O"],"axes":["N"]}
  ]
}
```

---

# 4. IndexBook（中核：軸・領域・アクセスの正規化サイドテーブル）

## 4.1 目的

* Tiny IR を汚さずに、**各値の**
  **(a) 軸（AxisVar）**、**(b) 領域（Domain）**、**(c) アクセス式（AxisMap）**、**(d) 縮約軸情報**
  を **決定的かつ機械変換可能な形**で保持する。

## 4.2 概念モデル（自然言語）

* **AxisVar**: 各値の**出力軸**。`i0,i1,…` の**連番命名**。`size ∈ {Int, Sym}`、`kind ∈ {iter, reduce, broadcast}`。
* **AxisExpr**: アフィンと **floordiv** のみ。剰余 `%` は **`E − q*floor(E/q)` + ガード**へ展開し保持しない。
* **AxisMap**: **出力軸 → 入力軸**の式ベクトル。値の各入力に 1 つずつ存在。
* **Domain**: 既定の範囲 `0 ≤ ik < sizek` に加え、`slice/pad/%` 由来条件は **piecewise** 化（複数ピースの合併）。
* **IndexInfo（値ごと）**: `axes / domain / inputs[{value_id,map}] / reduce_axes[] / flags(non_scop)` を持つ。
* **署名整列**: Frontend 署名のシンボル名を**唯一の真**とし、すべての値の Domain／AxisExpr を `align_params` で整列する。

## 4.3 全体不変（MUST）

1. あらゆる AxisMap は **アフィン + floordiv** のみ（`%` は禁止）。
2. 出力軸名は **`i0,i1,…` 連番**で、同名は同一空間の同一点。
3. ブロードキャストは式側の **定数0** で表現し、条件分岐を持たない。
4. Reduce は対象軸を出力から**削除**し、**削除前の軸ID**を `reduce_axes` に記録。
5. **非SCoP**（データ依存 index など）は `non_scop` でマークし、その点で **Region 境界**。
6. `floordiv` の分母は**正**。必要に応じ `gcd` で最簡形へ正規化。
7. Movement 連鎖は**逐次合成**し、`gist/coalesce/remove_redundancies` で式肥大を抑制。

## 4.4 オペレーション別 更新規則（準拠動作）

* **Leaf**（署名/テンソル表に対応）
  軸 `i0..` を発番し、`0 ≤ ik < Dk` を Domain に追加。`inputs=[]`。
* **Unary**
  **恒等継承**（軸/Domain/AxisMap そのまま）。
* **Binary（右端揃え Broadcast）**
  右端から形状を突き合わせ、両辺>1 の不一致で**即エラー**。
  出力軸を新設し、**一致側は恒等、放送側は 0** を AxisMap に設定。Domain は両入力の合流。
* **permute**
  置換行列で AxisMap を変換。
* **reshape/view**
  **線形化** `L = Σ ik*stridek` → **再分解** `oj = floor(L/stride’j) % size’j`。剰余由来の**ガード**を Domain に追加。
* **expand**
  新軸 `oj` を導入し AxisMap に **定数0**、Domain に `0 ≤ oj < new_size`。
* **slice/shrink**
  `lo ≤ ik < hi` を Domain に追加。`step>1` は `ik' = floor((ik−lo)/step)` とガードで表現。
* **pad**
  中央は恒等。境界は **別ピース** として Domain を分割（in‑bounds / out‑of‑bounds）。
* **Reduce(axes=K)**
  軸 `K` を `kind=reduce` にマークし**削除**。残軸は `i0..` で**名称再割当**（軸IDは保持）。`reduce_axes` に削除前IDを記録。

## 4.5 代表ダンプ（IndexBook 抜粋）

### GEMM（中間 T と縮約出力 C0）

```json
{
  "index_book": {
    "T": {
      "axes":[{"id":0,"name":"i0","size":"M","kind":"iter"},{"id":1,"name":"i1","size":"K","kind":"iter"},{"id":2,"name":"i2","size":"N","kind":"iter"}],
      "domain":{"constraints":[["0<=i0","i0<M"],["0<=i1","i1<K"],["0<=i2","i2<N"]]},
      "inputs":[{"value_id":"A","map":["i0","i1"]},{"value_id":"B","map":["i1","i2"]}]
    },
    "C0": {
      "axes":[{"id":3,"name":"i0","size":"M","kind":"iter"},{"id":4,"name":"i1","size":"N","kind":"iter"}],
      "domain":{"constraints":[["0<=i0","i0<M"],["0<=i1","i1<N"]]},
      "inputs":[{"value_id":"T","map":["i0","k","i1"]}],
      "reduce_axes":[1]
    }
  }
}
```

### Conv（stride=2, pad=1）— piecewise（in/out‑of‑bounds）

```json
{
  "index_book": {
    "Xv": {
      "axes":[{"id":0,"name":"n","size":"N","kind":"iter"},{"id":1,"name":"c","size":"C","kind":"iter"},{"id":2,"name":"h","size":"Ho","kind":"iter"},{"id":3,"name":"w","size":"Wo","kind":"iter"},{"id":4,"name":"kh","size":3,"kind":"iter"},{"id":5,"name":"kw","size":3,"kind":"iter"}],
      "domain":{"pieces":[
        {"constraints":[["0<=2*h+kh-1","2*h+kh-1<H+2*1"],["0<=2*w+kw-1","2*w+kw-1<W+2*1"]]},
        {"constraints":[["else_pad_piece"]]}
      ]},
      "inputs":[{"value_id":"Xp","map":["n","c","2*h+kh-1","2*w+kw-1"]}]
    }
  }
}
```

### Attention（多様マスクの broadcast 正規化）

```json
{
  "index_book": {
    "S_masked": {
      "axes":[{"id":0,"name":"b","size":"B","kind":"iter"},{"id":1,"name":"h","size":"H","kind":"iter"},{"id":2,"name":"m","size":"M","kind":"iter"},{"id":3,"name":"n","size":"N","kind":"iter"}],
      "inputs":[{"value_id":"S","map":["b","h","m","n"]},{"value_id":"mask","map":["b","h","m","n"]}]
    }
  }
}
```

## 4.6 IndexBook から得る解析量（下流への“意味”）

* **tail 軸**: タイル端で predication が必要な軸を抽出。
* **ハロー量**: pad/slice 起因の越境読取りを**軸別**（例：`h:±1, w:±1`）と**総Bytes**で算出。
* **最小バッファ種別**: `reg / smem_ring / gmem` と必要段数（2/3）。
* **compute‑at 合法性**: 生依存とハローから**融合可否**を判定。
  （結果は Poly Core で JSON レポートとして提供）

---

# 5. Poly‑View（解析専用IR）

## 5.1 役割と性質

* **保持対象**は **SCoP の Domain** と \*\*アクセス式（区分アフィン）\*\*のみ。
* `mul → reduce(sum)` を検出し、**ContractionPattern**（`matmul|conv`）として抽象化。
* `pad/slice/%` 由来境界は **piecewise domain**（union）で表現。
* **非アフィン**は SCoP から除外＝**その点で Region 境界**。

## 5.2 生成規則（IndexBook → Poly‑View）

* **空間正規化**: IndexBook の連番軸名をそのまま用い、余分な rename を作らない。
* **式の lowering**: AxisMap を `multi_aff / pw_multi_aff` へ。`floordiv` は `div` として導入。`%` 起源のガードは domain 側に。
* **簡約**: `coalesce / gist / remove_redundancies` を必須実行。
* **パターン検出**: `Binary(mul)` の直上 `Reduce(sum)` を候補化。二入力が Movement 以外を含まないことを確認し、共通軸／縮約軸を同定。

### 代表ダンプ（GEMM）

```json
{
  "poly_view": {
    "blocks": [
      {
        "name":"gemm_core",
        "kind":"contraction_pattern",
        "domain":{"set":"{ [m,n] : 0<=m<M and 0<=n<N }"},
        "accesses":[
          {"tensor":"A","map":"{ [m,n,k] -> A[m,k] }"},
          {"tensor":"B","map":"{ [m,n,k] -> B[k,n] }"},
          {"tensor":"C0","map":"{ [m,n] -> C0[m,n] }"}
        ],
        "attrs":{"pattern":"matmul","lhs_idx":["m","k"],"rhs_idx":["k","n"],"out_idx":["m","n"],"reduce_idx":["k"],"affine_offsets":{}}
      }
    ],
    "edges":[]
  }
}
```

---

# 6. Region Buffer SSA

## 6.1 役割と性質

* **Region ≒ 1 カーネル**。
* **出力のみ MemRef** を持ち、内部は値SSA。
* `mul+reduce + epilogue` は**同一 Region で融合**し、acc に直接適用。
* 出力 `materialize` は `deferred|gmem`（直後消費なら `deferred`）。

### 代表ダンプ（GEMM+Epilogue 融合）

```json
{
  "region": {
    "name":"gemm_bias_relu",
    "inputs":[
      {"name":"A","memref":{"shape":["M","K"],"alignment":128,"noalias":true}},
      {"name":"B","memref":{"shape":["K","N"],"alignment":128,"noalias":true}},
      {"name":"bias","memref":{"shape":["N"],"alignment":128,"noalias":true}}
    ],
    "outputs":[{"name":"C","memref":{"shape":["M","N"],"alignment":128,"noalias":true},"materialize":"gmem"}],
    "body":[
      {"let":"C0","op":{"kind":"contraction","pattern":"matmul","lhs":"A","rhs":"B","acc_dtype":"fp32"}},
      {"let":"C1","op":{"kind":"ewise","fn":"add","inputs":["C0","bias"]}},
      {"let":"C2","op":{"kind":"unary","fn":"relu","inputs":["C1"]}},
      {"yield":["C2 -> C"]}
    ]
  }
}
```

---

# 7. Poly Core（多面体解析サービス）

## 7.1 提供情報

* **依存**（True/Anti/Output）
* **並列可能軸／縮約軸**
* **アフィン閉性**（合成後も区分アフィンか）
* **tail 軸抽出**（predication 対象）
* **バリア挿入ヒント**（SM80: commit/wait, SM90: mbarrier）
* **バンク衝突ヒント**（スウィズル推奨）
* **compute‑at 合法性**（ハロー量含む）
* **最小バッファ**（`reg / smem_ring / gmem`, depth=2|3）

### レポート例

```json
{
  "analysis": {
    "parallel_axes":["m","n"],
    "reduce_axes":["k"],
    "tail_axes":["m","n","k"],
    "compute_at":{"ok":true,"halo":{"bytes":0,"per_axis":{"m":0,"n":0,"k":0}}},
    "min_buffer":{"buffer":"reg","depth":2}
  }
}
```

---

# 8. Schedule Plan（JSON 定義と意味）

## 8.1 フィールドとセマンティクス

* `tile=[BM,BN,BK]` … CTA タイル
* `stages ∈ {2,3}` … SMEM パイプライン段数
* `bind` … 軸の CTA/warp/lane への割付（例：`m.o→block.y`）
* `warp_tile` … warp 内 MMA 形状（例：`64x64`）
* `cache` … どのテンソルをどこでキャッシュ（`smem`）し、どの軸で入替（`at`）
* `vectorize` … グローバル ld/st のベクトル幅
* `predicate_tail` … predication 対象軸
* `epilogue` … acc 直接適用オペ（bias/activation）
* `arch ∈ {"sm80","sm90"}` … アーキ分岐
* `layout_hints` … スウィズルや出力ストライド指示
* `algo_choice` … `matmul|conv|attention` の実装分岐
* `local_edges` … 値間前渡し（`reg`）や **SMEM リング**（`smem_ring`, depth）

### 代表ダンプ（GEMM/SM80）

```json
{
  "tile":[128,64,64],
  "stages":2,
  "bind":{"m.o":"block.y","n.o":"block.x","m.i.o":"warp.y","n.i.o":"warp.x"},
  "warp_tile":"64x64",
  "cache":[
    {"tensor":"A","where":"smem","at":"k.i","pingpong":true},
    {"tensor":"B","where":"smem","at":"k.i","pingpong":true}
  ],
  "vectorize":{"axis":"n.i.i","width":8},
  "predicate_tail":["m.i.i","n.i.i","k.i.i"],
  "epilogue":["bias","relu"],
  "arch":"sm80",
  "layout_hints":{"A_swizzle":true,"B_swizzle":true,"C_stride":"row"},
  "algo_choice":{"matmul":"implicit_gemm"},
  "local_edges":[{"from":"sum_k","to":"relu","buffer":"reg"}]
}
```

---

# 9. GPU方言IR（単一MMAテンプレート）

## 9.1 固定骨格と意味

* **SM80**: `cp.async → commit/wait → ldmatrix → mma.sync → epilogue → st.global.vN`
* **SM90**: `tma.load → mbarrier.arrive/wait → wgmma.mma_async → epilogue → st.global.vN`
* **アーキ分岐**は本層のみ。
* **tail** は Plan の `predicate_tail` に従い predicated。
* **Epilogue** は **acc** に直適用（GMEM 往復禁止）。

### 代表ダンプ（SM90）

```json
{
  "gpu_ir": {
    "arch":"sm90",
    "stmts":[
      {"TmaLoad":{"dst_smem":"smA[s0]","desc":"A_desc","coords":"(m,k0)","mbarrier":"mb"}},
      {"TmaLoad":{"dst_smem":"smB[s0]","desc":"B_desc","coords":"(k0,n)","mbarrier":"mb"}},
      {"MBarrierArrive":{"mbarrier":"mb"}},
      {"MBarrierWait":{"mbarrier":"mb"}},
      {"Wgmma":{"acc":"acc","a_desc":"smA_tile","b_desc":"smB_tile","shape":"64x128x64","dtype":"fp16_fp32acc"}},
      {"Epilogue":{"acc":"acc","ops":["bias"]}},
      {"StGlobalVec":{"ptr":"C+coff","regs":"acc_vec","pred":"p_tail"}}
    ]
  }
}
```

---

# 10. 変換パイプライン（Front→Tiny→IndexBook→Poly‑View→Region→Plan→GPU）

1. **Frontend→Tiny**
   高位演算を最小語彙に正規化。`Conv2D` は `pad + movement(affine) + mul + reduce`。`Attention` は 2 回の `mul+reduce` と行 softmax。
2. **Tiny→IndexBook**
   値ごとに **Axis/Domain/AxisMap** を導出。`reshape/view` は**線形化→再分解**。`%` は**div+ガード**。
3. **ContractionPattern 検出**
   `Binary(mul)` 直上に `Reduce(sum)` を確認。二入力が Movement 以外を含まないこと。軸比較で共通・縮約軸を同定。
4. **IndexBook→Poly‑View**
   SCoP のみ抽出し、アクセスを `pw_multi_aff` に落とす。`pieces` は `union` 化。
5. **Region 生成**
   `can_compute_at` と**ハロー量**で融合可否を判定。**出力のみ MemRef**、中間は値SSAのまま。
6. **Plan 付与**
   解析ヒント（並列軸／最小バッファ／バリア）を利用し探索空間を 10〜20 点に絞る。
7. **GPU IR 生成**
   Plan をテンプレに注入。アーキ分岐と tail predication を実装。

---

# 11. コストモデルとチューニング

* **基本式**
  `time_est = max( FLOPs/(peak_TF*occ*pipeline_eff),  Bytes/(peak_BW*occ) )`
* **融合の差分評価**
  `Δtime = max(ΔFLOPs/(peak_TF*occ*pipeline_eff), ΔBytes/(peak_BW*occ)) + Δocc_time + penalties − saved_bytes/peak_BW`
  `penalties = fanout_penalty + layout_penalty(TMA/WGMMA, vectorize)`
* **探索空間（MVP）**: `tile(3–4) × stages{2,3} × warp_tile{2} × vec{4,8,16} × algo_choice（少数）`
* **STOP 条件**

  * `smem_per_CTA` 推定が上限の **\~80%** 超
  * `regs_per_thr` 増で **CTA/SM=1** に低下
  * `vectorize.width` が **16→8/4** に縮小
  * **SM90**: TMA/WGMMA 要求タイルから外れる

---

# 12. ランタイム／ABI／ShapeSets

* **Graph Signature** を ABI として採用（I/O 役割・可変性・アライン）。
* **ConstPool**: weights の連続配置と整列。
* **Memory Arena**: 生存区間解析に基づく再利用。
* **ShapeSets**: 動的形状をクラスタ化し Plan/実測をキャッシュ。
* **CUDA Graph**: 形状クラスタごとに 1 個キャプチャし再利用。
* **RNG（学習）**: counter‑based（Philox系）で副作用制御。
* **エラー**: SMEM/REG 超過・非アフィン・アライン不整合は**明示診断**。

---

# 13. デバッグ／検証

* **ダンプ**: `--dump=frontend,tiny,indexbook,poly_view,region,plan,gpu,cu`
* **CPU プレイバック**: Region 内算術を JIT/解釈で実行し数値検証。
* **精度検証**: FP32 参照と `rtol=1e-3, atol=1e-3` で比較。
* **測定ログ**: `FLOPs/Bytes/occ/estimate/actual/plan` を CSV 追記。

---

# 14. 失敗条件と診断（Index/isl まわりの規範）

* **Broadcast 不一致**: 右端揃えで size>1 の次元が一致しない → **Tiny 構築時に停止**し左右形状を提示。
* **負ストライド view**: 未対応 → **Movement 段で拒否**。
* **`%` 展開のガード欠落**: `reshape/view` で `%` を保持してしまった → **Poly‑View 生成前に検証で停止**。
* **piecewise 爆発**: `pad/slice` 多用で union が過大 → `gist/coalesce/remove_redundancies` を**強制**。
* **非SCoP**: データ依存 index → **その点で Region 境界確定**。
* **資源超過**: `smem_per_CTA` / `regs_per_thr` 超過予測 → Plan を縮退（`stages/warp_tile/vec` を下げる）。

### 代表診断ダンプ

```json
{
  "diagnostics":[
    {"kind":"BroadcastMismatch","at_op":"Binary(mul)","lhs_shape":["M","K","N"],"rhs_shape":["M","K","N'"]},
    {"kind":"NegativeStrideUnsupported","at_op":"Movement(view)"},
    {"kind":"MissingDivGuard","at":"Poly-View"},
    {"kind":"NonSCoP","at_value":"gather_idx","note":"region boundary forced"}
  ]
}
```

---

# 15. 端から端までの例（抜粋リンク）

## 15.1 GEMM + Bias + ReLU

* Frontend 署名＋DAG（§2.2）
* Tiny（§3）
* IndexBook（§4.5）
* Poly‑View（§5.2）
* Region（§6）
* Plan（§8.1 例）
* GPU IR（§9）

## 15.2 Conv 3×3 + SiLU

* Tiny: `pad + movement(affine) + mul + reduce`
* IndexBook: **piecewise** と **ハロー ±1**
* Poly‑View: `pattern="conv"`（`affine_offsets` に `(h+kh-1, w+kw-1)`）
* Region: `local_edges=smem_ring`
* Plan: `stages=3, warp_tile=64x32, vec=8`
* GPU IR: SM80 骨格

## 15.3 Attention（causal）

* Tiny: QKᵀ→softmax→V（2 contraction + reduce/ewise）
* IndexBook: mask の多様ブロードキャストを 0 写像で正規化
* Poly‑View: `qk_matmul`／`row_softmax`／`pv_matmul`
* Region: `P` を `materialize:"deferred"`、2 Region 連結
* Plan: `"attention":"streaming_softmax_2pass"`（SM90 推奨）
* GPU IR: `tma + mbarrier + wgmma`、tail は行/列で predicated

---

## 付記：本改訂の要点（変更点の明確化）

* **Graph Signature（I/O）を“必須”として明記**（演算ノード化しない）
* **IndexBook を中心**に、データモデル・不変条件・更新規則・piecewise/floordiv 取り扱いを**厳密化**
* 各層に**JSON ダンプ例**を多数追加（GEMM/Conv/Attention）
* **Plan→GPU IR** の対応関係とアーキ分岐/Tail/Epilogue の**規範化**
* 失敗条件と診断出力の**具体化**

