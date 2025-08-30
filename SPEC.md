
# 深層学習コンパイラ 仕様書 

## 0. スコープと原則（不変）

* 対象 GPU: **NVIDIA SM80（Ampere）/ SM90（Hopper） 以降**
* 主精度: **FP16 / BF16**（必要箇所で **FP32 蓄積**）
* 外部ライブラリ: 不使用（**CUDA/C 自動生成**、必要箇所のみ inline PTX）
* 性能方針: **単一MMAテンプレート**に **Schedule Plan** を注入。分岐は **アーキ（SM80/SM90）** と **tail** のみに限定
* MVP: **GEMM / Conv / Attention**（Epilogue 融合を原則）

### 0.1 用語（本仕様の定義）

* **ハロー（halo）**: 出力タイル計算に必要な入力側の追加読取り“縁”
* **ステージ（pipeline stages）**: 非同期ロードと演算の重ね段数（2=二重、3=三重）
* **fan‑out**: ある値の出力を複数の下流が消費（DAG出次数>1）
* **SCoP**: 添字・境界が（区分）アフィンで表せる部分
* **UCC**: 単一消費者鎖（分岐しない下流パス）

---

# 1. アーキ概要と責務分離

本コンパイラは「**意味（What）**」「**添え字と領域（Where）**」「**実装（How）**」を厳密に分離します。

1. **Frontend IR** … モデルの意味（高位演算）。\*\*グラフ署名（I/O）\*\*を持つ
2. **Tiny IR** … **Movement / Unary / Binary / Reduce**の最小語彙で純関数DAG化
3. **IndexBook（中核）** … 各値の**軸（Axis）・領域（Domain）・アクセス式（AxisMap）**をサイドテーブルで**正規化**
4. **Poly‑View** … **SCoP とアクセス式のみ**を保持。`mul→reduce(sum)` を **ContractionPattern**に抽象化
5. **Region Buffer SSA** … **1 Region = 1 カーネル**。出力のみ MemRef、内部は値SSA。**明示 Broadcast/Transpose**と**SRNF**で正規化
6. **Poly Core** … 依存解析／ハロー算出／最小バッファ分類／バリア・バンク衝突ヒント
7. **Schedule Plan（JSON/DSL）** … タイル・割付・パイプライン・非同期化等の方針
8. **GPU方言IR** … 固定骨格（`cp.async|tma → smem ping‑pong → ldmatrix|wgmma → epilogue → st.global`）に Plan を注入。**非同期トークン**で依存を表現

**設計原則**

* **非アフィン**は Poly‑View から除外し、そこで **Region 境界**を確定
* \*\*物質化（gmem ラウンドトリップ）\*\*は最小化。Epilogue は **acc 直接適用**
* **Front→Tiny→IndexBook**の流れは**決定的**で再現可能

---

# 2. Frontend IR（高位演算）

## 2.1 役割

* Conv / Attention / LayerNorm / Elementwise / Reduce 等の**高位属性**（例：stride, pad, heads, axis, eps）を保持
* **実メモリコピー**は行わない（Movement は論理ビューのみ）

## 2.2 グラフ署名（I/O）— 必須

* **MUST**: **Graph Signature** として、モデルの **入力/出力**を明示
* 署名は ABI・ShapeSets・最適化境界の**基準**
* `Input/Output` を**演算ノード化しない**（デバッグ用仮ノードは MAY）

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

* すべてのノードは**純関数**。副作用なし
* 形状は整数と記号（Sym）で表現。未確定は **ShapeSets** でクラスタ化
* Epilogue 候補は**融合可能**な位置に置く

---

# 3. Tiny IR（**tinygrad UOp 互換**）

> 目的：本層は **tinygrad の UOp グラフ**に準拠し、\*\*view（ShapeTracker系）**と**base（計算 / バッファ境界）\*\*の二系統で表現します。UOp は `op, dtype, src[], arg, tag` を持つノードで構成されます。base は連続バッファへ計算、view は ShapeTracker による論理ビューを表します。([docs.tinygrad.org][1])
> UOp の代表オペ群（`Ops`）には VIEW 系（`RESHAPE/PERMUTE/EXPAND/PAD/SHRINK/FLIP/VIEW/VALID`）、算術（`ADD/MUL/SUB/.../WHERE`）、縮約（`REDUCE/ALLREDUCE/REDUCE_AXIS`）、ロード/ストア（`LOAD/STORE/ASSIGN`）、収束/収縮（`CONTRACT/WMMA`）、制御補助（`RANGE/IF/ENDIF` ほか）が含まれます。([docs.tinygrad.org][2])

## 3.1 語彙（本仕様で採用する **最小サブセット**）

**View UOps（Movement / 形状・添字ビュー）**

* `VIEW, RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP, VALID`

  * **意味**: 実メモリ移動なし。IndexBook の AxisMap/Domain に反映（§4）。
  * **放送**は `EXPAND` で、**右端揃え**互換の形状合一を表現。

**Base UOps（計算・縮約・I/O）**

* **Elementwise/算術**: `ADD, MUL, SUB, NEG, MAX, WHERE, FDIV, POW, ...`
* **Reduce**: `REDUCE(op=SUM|MAX|MIN, axes=...)`（keepdim 無しの規約）
* **Contraction**: `CONTRACT`（`mul→reduce(sum)` パターンを 1UOp に抽象化、後工程で `Reduce→MMA/WGMMA` に tensorize）
* **Load/Store**: `LOAD(tensor, map)` / `STORE`（Region 生成時に `Read/Select/Reduce`へ正規化）
* **Index/制御**: `INDEX, RANGE, IF/ENDIF`（内部表現。非本質部は Region 生成時に predication へ）

> **注**: tinygrad 既存の `BUFFER/DEFINE_*` 等の UOp は**本仕様では Region 以降**で扱う（RB‑SSAで MemRef として具現化）。Tiny IR 段階では \*\*base UOp の“計算意味”\*\*のみを用いる。([docs.tinygrad.org][2])

## 3.2 旧「Movement/Unary/Binary/Reduce」との **1:1 対応**

* **Movement** → **View UOps**（`RESHAPE/PERMUTE/EXPAND/PAD/SHRINK/FLIP/VIEW/VALID`）
* **Unary/Binary** → **算術 UOps**（`NEG/SQRT/EXP2/ADD/MUL/...`）
* **Reduce** → **`REDUCE`**（軸削除の規約は維持）
* **mul→reduce(sum)** → **`CONTRACT`**（または `MUL`＋`REDUCE(SUM)` の素朴形）

## 3.3 セマンティクス（本仕様）

* **View**: ShapeTracker で表現。**実体化なし**。IndexBook の Domain は piecewise 可能（`PAD/SHRINK` 等）。
* **Elementwise**: 右端揃えブロードキャスト。必要な `EXPAND` は **自動挿入**（IndexBookで検証）。
* **Reduce**: 指定軸を出力から**削除**（keepdim=false）。Neutral は op ごとに定義（SUM→0, MAX→-INF 等）。
* **Contraction**: `CONTRACT(lhs_idx, rhs_idx, out_idx, reduce_idx)` の属性を持つ高位 UOp（Poly‑View で ContractionPattern と整合）。
* **Load/Store**: Frontend I/O の**論理アクセス**を明示するための記号的 UOp。**Region 生成時**に `Read/Select` 系へ落とし込み（§6）。

## 3.4 エラー条件（更新）

* **BroadcastMismatch**（両辺 size>1 不一致）— 旧仕様どおり
* **非アフィン View**（IndexBook 非SCoP）— Region 境界確定
* **NegativeStride** は **`FLIP`** で表現可能のため **エラーにしない**（旧 E1102 を撤回）。([docs.tinygrad.org][2])
* `slice` の `step>1` は `floordiv+ガード` で正規化不能な場合のみエラー

## 3.5 代表ダンプ（UOp 互換ダンプ）

> **表記**: `{"uop":"OPS.NAME","src":[...],"arg":{...}}`。簡潔化のため dtype/tag は省略。

### 3.5.1 GEMM + Bias + ReLU（C‑CON1 準拠）

```json
{
  "uops": [
    {"uop":"VIEW","src":["A"]},
    {"uop":"VIEW","src":["B"]},
    {"uop":"CONTRACT","arg":{"pattern":"matmul","lhs_idx":["m","k"],"rhs_idx":["k","n"],"out_idx":["m","n"],"reduce_idx":["k"]},"src":["A","B"],"out":"Cacc_fp32"},
    {"uop":"EXPAND","src":["bias"],"arg":{"result_shape":["M","N"],"broadcast_dimensions":[1]},"out":"biasMN"},
    {"uop":"ADD","src":["Cacc_fp32","biasMN"],"out":"C1"},
    {"uop":"WHERE","src":[{"gt":"C1,0"}, "C1", 0.0],"out":"C2_fp32"},
    {"uop":"CAST","src":["C2_fp32"],"arg":{"to":"fp16"},"out":"C2"}
  ]
}
```

### 3.5.2 Conv 3×3（stride=2, pad=1）

```json
{
  "uops": [
    {"uop":"PAD","src":["X"],"arg":{"pad": [0,0,1,1,1,1]},"out":"Xp"},
    {"uop":"VIEW","src":["Xp"],"out":"Xv"}, 
    {"uop":"CONTRACT","arg":{"pattern":"conv2d","lhs_idx":["co","ci","kh","kw"],"rhs_idx":["n","ci","ho","wo","kh","kw"],"out_idx":["n","co","ho","wo"],"reduce_idx":["ci","kh","kw"],"strides":[2,2]},"src":["W","Xv"],"out":"Yacc_fp32"},
    {"uop":"CAST","src":["Yacc_fp32"],"arg":{"to":"fp16"},"out":"Y"}
  ]
}
```

### 3.5.3 Attention（row softmax まで）

```json
{
  "uops":[
    {"uop":"CONTRACT","arg":{"pattern":"matmul","lhs_idx":["b","h","m","d"],"rhs_idx":["b","h","n","d"],"out_idx":["b","h","m","n"],"reduce_idx":["d"]},"src":["Q","K"],"out":"S"},
    {"uop":"REDUCE","src":["S"],"arg":{"op":"MAX","axes":["n"]},"out":"Mrow"},
    {"uop":"SUB","src":["S","Mrow"],"out":"S0"},
    {"uop":"EXP2","src":[{"mul":["S0",1.442695]}],"out":"A"},  // exp(x)=2^{x*log2(e)}
    {"uop":"REDUCE","src":["A"],"arg":{"op":"SUM","axes":["n"]},"out":"Z"},
    {"uop":"FDIV","src":["A","Z"],"out":"P_fp32"},
    {"uop":"CAST","src":["P_fp32"],"arg":{"to":"fp16"},"out":"P"}
  ]
}
```

> **備考**: tinygrad の UOp 群と **base/view の二分**は公式開発者ドキュメントの記述と整合します。([docs.tinygrad.org][1])

---

# 4. IndexBook（中核：軸・領域・アクセスの正規化サイドテーブル）

## 4.1 目的

Tiny IR を汚さずに、**各値**の
(a) **軸（AxisVar）**、(b) **領域（Domain）**、(c) **アクセス式（AxisMap）**、(d) **縮約軸情報**
を**決定的**・**機械変換可能**な形で保持します。

## 4.2 概念モデル

* **AxisVar**: 各値の**出力軸**。`i0,i1,…` と連番命名。`size ∈ {Int, Sym}`、`kind ∈ {iter, reduce, broadcast}`
* **AxisExpr**: アフィン + **floordiv** のみ（`%` は `E − q*floor(E/q)` + ガードで表現）
* **AxisMap**: **出力軸 → 入力軸**の式ベクトル。値の各入力に 1 つ存在
* **Domain**: 既定範囲 `0 ≤ ik < sizek` に加え、`slice/pad/%` 由来条件は **piecewise** 化（複数ピース合併）
* **IndexInfo（値ごと）**: `axes / domain / inputs[{value_id,map}] / reduce_axes[] / flags(non_scop)`
* **署名整列**: Frontend 署名のシンボル名を**唯一の真**とし、全値の Domain／AxisExpr を `align_params` で整列

## 4.3 全体不変（MUST）

1. AxisMap は **アフィン + floordiv** のみ（`%` 禁止）
2. 出力軸名は **`i0,i1,…` 連番**で、同名は同一空間の同一点
3. ブロードキャストは式側の **定数0** で表現し、条件分岐を持たない
4. Reduce は対象軸を出力から**削除**し、**削除前の軸ID**を `reduce_axes` に記録
5. **非SCoP**は `non_scop` でマークし、そこで **Region 境界**
6. `floordiv` の分母は**正**。必要時 `gcd` で最簡形へ正規化
7. Movement 連鎖は**逐次合成**し、`gist/coalesce/remove_redundancies` で式肥大を抑制

## 4.4 オペレーション別 更新規則（準拠動作）

* **Leaf**: 軸 `i0..` を発番し `0 ≤ ik < Dk` を Domain に追加。`inputs=[]`
* **Unary**: **恒等継承**
* **Binary（右端揃え Broadcast）**: 右端から形状を突き合わせ、両辺>1 の不一致で**即エラー**。出力軸を新設し、一致側は恒等、放送側は 0 を AxisMap に設定。Domain は両入力の合流
* **permute**: 置換行列で AxisMap を変換
* **reshape/view**: **線形化** `L = Σ ik*stridek` → **再分解** `oj = floor(L/stride’j) % size’j`。`%` 由来ガードを Domain に追加
* **expand**: 新軸 `oj` を導入し AxisMap に **定数0**、Domain に `0 ≤ oj < new_size`
* **slice/shrink**: `lo ≤ ik < hi` を Domain に追加。`step>1` は `ik' = floor((ik−lo)/step)` とガード
* **pad**: 中央は恒等。境界は **別ピース** として Domain 分割（in‑bounds / out‑of‑bounds）
* **Reduce(axes=K)**: 軸 `K` を `kind=reduce` にマークし**削除**。残軸は `i0..` で**名称再割当**（軸IDは保持）。`reduce_axes` に削除前IDを記録

## 4.5 代表ダンプ（抜粋）

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

## 4.6 IndexBook から得る解析量

* **tail 軸**: タイル端で predication が必要な軸を抽出
* **ハロー量**: pad/slice 起因の越境読取りを**軸別**と**総Bytes**で算出
* **最小バッファ種別**: `reg / smem_ring / gmem` と必要段数（2/3）
* **compute‑at 合法性**: 生依存とハローから**融合可否**を判定（Poly Core で JSON レポート化）

---

# 5. Poly‑View（解析専用IR）

## 5.1 役割と性質

* **保持対象**は **SCoP の Domain** と \*\*アクセス式（区分アフィン）\*\*のみ
* `mul → reduce(sum)` を検出し、**ContractionPattern**（`matmul|conv`）として抽象化
* `pad/slice/%` 由来境界は **piecewise domain**（union）で表現
* **非アフィン**は SCoP から除外＝**その点で Region 境界**

## 5.2 生成規則（IndexBook → Poly‑View）

* **空間正規化**: IndexBook の連番軸名を踏襲し rename を抑制
* **式の lowering**: AxisMap を `multi_aff / pw_multi_aff` へ。`floordiv` は `div` として導入。`%` 起源ガードは domain 側へ
* **簡約**: `coalesce / gist / remove_redundancies` を必須実行
* **パターン検出**: `Binary(mul)` 直上 `Reduce(sum)` を候補化。入力が Movement 以外を含まないこと。共通軸／縮約軸を同定

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

# 6. Region Buffer SSA（改訂：明示 Broadcast/Transpose + SRNF + レイアウト属性）

## 6.1 役割と境界

* **目的**: IndexBook で確定した **軸・領域（Where）**を、**非物質化の最小語彙**で**純関数 SSA**に落とし、以降の Plan/GPU 層が**推測ゼロ**で実装判断できる形にする
* **Region = 1 カーネル**（不変）。Region 内は**副作用なし**。**出力のみ MemRef**（`yield` で書出し）
* **語彙（最小集合）**
  **Read / Unary / Binary / Select / Reduce / BroadcastInDim / Transpose**
* **非目標**: 高位パターン名（matmul/conv/attention）を **RB‑SSA には持ち込まない**。パターン検知は Poly‑View（解析専用）

## 6.2 値モデル・MemRef レイアウト（新規）

* `MemRef{shape, alignment, noalias, layout}` を**明示**

  * `layout.kind ∈ {"row_major","col_major","swizzled"}` 等
  * `layout.params` にスウィズル・ストライド・パディング等の可変パラメータ（例：`{"swizzle":"xor32","vec_bytes":16}`）
* **意味**

  * `Transpose` は**論理ビュー**（非物質化）として RB‑SSA に現れる
  * **Plan/GPU IR** は `layout` と `Transpose` を読み、`ldmatrix(.trans)` や **TMA desc** のストライド設定で実現
  * 例外的に物理レイアウト変換を要する場合は **Region 境界**で materialize を選択（Plan の裁量）

## 6.3 式（Expr）

* `Read(tensor, map)` … AxisMap（アフィン+floordiv）で **アドレスのみ**決まる論理読取り
* `Unary(fn, x)` / `Binary(fn, x, y)` … 要素演算。**Binary 直前に型検査**し必要なら `BroadcastInDim` を**自動挿入**
* `Select(pred, then, else)` … piecewise/pad/tail を **predication** で表現（Reduce の neutral は演算ごとに定義）
* `Reduce(op, axes, init, dtype, body)` … `op ∈ {sum,max,min}`。`axes` は IndexBook の `reduce_axes` と整合。`dtype` は蓄積型
* `BroadcastInDim(operand, result_shape, broadcast_dimensions[])` … **非物質化**のブロードキャスト
* `Transpose(operand, permutation[])` … **非物質化**の転置（後段で `ldmatrix.trans` / TMA desc に落ちる）

### 6.3.1 収束規約（Canonical Contraction：新規）

* **C‑CON1（推奨）**: `Reduce` の `body` 内では **ブロードキャスト済みテンソルを作らず**、**必要な添字で `Read` を直接行う**

  * 例：GEMM は `mul( Read(A[m,k]), Read(B[k,n]) )` を `sum(k)` で縮約
  * これにより `tensorize`（Reduce→MMA/WGMMA）のパターン化が直感的になり、生成器が簡素化

## 6.4 型規則とブロードキャスト（明示化）

1. **Binary 型検査（MUST）**
   右端揃えルールで突き合わせ、必要時 **`BroadcastInDim` を自動挿入**。両辺 size>1 不一致は **`BroadcastMismatch`**
2. **Transpose（SHOULD）**
   添字順序の交換は **`Transpose` を明示**。下層は `permutation[]` と `layout` を読むだけ
3. **present\_axes（MAY）**
   実装便宜上、各値に `present_axes`（自由軸集合）を保持可。ただし最終的に **IR 上に `BroadcastInDim`/`Transpose` が存在**すること

## 6.5 Movement 正規化（非物質化）

* `permute/transpose` → `Transpose`
* `expand/broadcast` → `BroadcastInDim`
* `reshape/view` → `Read` の AxisMap へ折込み（必要なら `BroadcastInDim` 併用）
* `pad/slice/step/floordiv/%` → `Select(pred, then, neutral)`
* **非SCoP** → `non_scop` として **Region 境界**

## 6.6 Reduce 正規化（SRNF）と合法変形

* **R0: 合流** `Reduce(op,K1){ Reduce(op,K2){E} } → Reduce(op,K1∪K2){E}`
* **R1: 不変 Hoist** `Depends(g,K)=false` ⇒ `g(Reduce(op,K){E})`
* **R2: 分配（任意）** `sum` の加法/スカラーなど安全な場合のみ
* **R3: Post‑Ewise（Epilogue）** `Depends(post,K)=false` な ewise は Reduce **外**に置き acc へ直適用
* **R4: 異種 op バリア** `max`→`exp`→`sum` 等は合流しない
* **SRNF**: 可能な限り `let acc = Reduce(op,K){body(K)}` へ単一化

## 6.7 Region 合成手順（IndexBook→RB‑SSA）

1. **Axis 整列**: IndexBook の連番軸で `iters` を確定
2. **Movement 正規化**: `BroadcastInDim`/`Transpose`/`Read`/`Select` に変換
3. **Binary 型検査**: 必要に応じ **`BroadcastInDim` 自動挿入**
4. **Reduce 解析**: `Depends(·, axes)` を付与。R1→R0→必要なら R2 を適用し SRNF 化
5. **Select 正規化**: 境界は **Reduce の内側**で neutral を選ぶ形に保持（unguarded 禁止）
6. **SSA 共有**: Reduce 結果の fan‑out は再計算禁止
7. **Epilogue**: post‑ewise を Reduce 後に直列化（acc へ直適用）
8. **Yield**: 出力のみ materialize。`"deferred"` は直後 Region へ前渡し（GMEM 往復禁止ヒント）

## 6.8 代表ダンプ（MVP）

### 6.8.1 GEMM + Bias + ReLU（**C‑CON1 遵守版**）

```json
{
  "region": {
    "name": "gemm_bias_relu",
    "iters": [{"name":"m","size":"M"},{"name":"n","size":"N"}],
    "inputs": [
      {"name":"A","memref":{"shape":["M","K"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}}},
      {"name":"B","memref":{"shape":["K","N"],"alignment":128,"noalias":true,"layout":{"kind":"col_major"}}},
      {"name":"bias","memref":{"shape":["N"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}}}
    ],
    "outputs": [{"name":"C","memref":{"shape":["M","N"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}},"materialize":"gmem"}],
    "lets": [
      {"let":"acc","expr":{"reduce":{"op":"sum","axes":["k"],"init":0.0,"dtype":"fp32",
        "body":{"mul":[
          {"read":{"tensor":"A","map":["m","k"]}},
          {"read":{"tensor":"B","map":["k","n"]}}
        ]}}}},
      {"let":"c1","expr":{"add":["acc", {"BroadcastInDim":{
        "operand":{"read":{"tensor":"bias","map":["n"]}},
        "result_shape":["M","N"], "broadcast_dimensions":[1]
      }}]}},
      {"let":"c2","expr":{"relu":["c1"]}}
    ],
    "yield":[{"from":"c2","to":"C","cast":"fp16"}]
  }
}
```

> **注**: A/B の転置は **MemRef の `layout`** と **下流の `ldmatrix.trans`/TMA desc** で吸収します。

### 6.8.2 Conv 3×3（stride=2, pad=1；単一 Region）

```json
{
  "region": {
    "name":"conv3x3_s2_p1",
    "iters":[{"name":"n"},{"name":"co"},{"name":"ho"},{"name":"wo"}],
    "inputs":[
      {"name":"X","memref":{"shape":["N","Ci","H","W"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}}},
      {"name":"W","memref":{"shape":["Co","Ci",3,3]  ,"alignment":128,"noalias":true,"layout":{"kind":"row_major"}}}
    ],
    "outputs":[{"name":"Y","memref":{"shape":["N","Co","Ho","Wo"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}},"materialize":"gmem"}],
    "lets":[
      {"let":"y_acc","expr":{"reduce":{"op":"sum","axes":["ci","kh","kw"],"init":0.0,"dtype":"fp32",
        "body":{"mul":[
          {"read":{"tensor":"W","map":["co","ci","kh","kw"]}},
          {"select":{
            "pred":{"and":[
              {"ge":[{"add":[{"mul":[2,"ho"]},{"add":["kh",-1]]}],0]},
              {"lt":[{"add":[{"mul":[2,"ho"]},{"add":["kh",-1]]}],{"add":["H",2]}}],
              {"ge":[{"add":[{"mul":[2,"wo"]},{"add":["kw",-1]]}],0]},
              {"lt":[{"add":[{"mul":[2,"wo"]},{"add":["kw",-1]]}],{"add":["W",2]}}]
            ]},
            "then":{"read":{"tensor":"X","map":["n","ci",
              {"add":[{"mul":[2,"ho"]},{"add":["kh",-1]]}],
              {"add":[{"mul":[2,"wo"]},{"add":["kw",-1]]}]
            ]}},
            "else":0.0
          }}
        ]}}}}
    ],
    "yield":[{"from":"y_acc","to":"Y","cast":"fp16"}]
  }
}
```

### 6.8.3 Attention（行 softmax まで；異種 Reduce は合流しない）

```json
{
  "region": {
    "name":"attention_row_softmax",
    "iters":[{"name":"b"},{"name":"h"},{"name":"m"},{"name":"n"}],
    "inputs":[
      {"name":"Q","memref":{"shape":["B","H","M","D"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}}},
      {"name":"K","memref":{"shape":["B","H","N","D"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}}}
    ],
    "outputs":[{"name":"P","memref":{"shape":["B","H","M","N"],"alignment":128,"noalias":true,"layout":{"kind":"row_major"}},"materialize":"deferred"}],
    "lets":[
      {"let":"S","expr":{"reduce":{"op":"sum","axes":["d"],"init":0.0,"dtype":"fp32",
        "body":{"mul":[
          {"read":{"tensor":"Q","map":["b","h","m","d"]}},
          {"read":{"tensor":"K","map":["b","h","n","d"]}}
        ]}}}},
      {"let":"Mrow","expr":{"reduce":{"op":"max","axes":["n"],"init":"-INF","dtype":"fp32","body":"S"}}},
      {"let":"A","expr":{"exp":[{"sub":["S","Mrow"]}]}},
      {"let":"Z","expr":{"reduce":{"op":"sum","axes":["n"],"init":0.0,"dtype":"fp32","body":"A"}}},
      {"let":"Pval","expr":{"div":["A","Z"]}}
    ],
    "yield":[{"from":"Pval","to":"P","cast":"fp16"}]
  }
}
```

## 6.9 中間データとバッファ（責務分離）

* RB‑SSA は**値SSA**のみを表し、\*\*内部バッファ確保（reg/smem）・段数（2/3）\*\*は記述しない
* **最小バッファ種別**（`reg / smem_ring / gmem`, depth）は **Poly‑Core** と **Plan**で決定
* `materialize:"deferred"` は **同一 SM 内の直後 Region** に前渡し（GMEM 往復禁止ヒント）

## 6.10 診断（エラー/警告）

* **BroadcastMismatch**
* **AxisAlignmentMismatch**
* **UnguardedAccess**
* **AccDtypeMissing**
* **NonAssociativeTransform**
* **NeutralInvalid**
* **ExcessivePiecewise**

（詳細・対処は §14 参照）

## 6.11 検証とプレイバック

* **CPU プレイバック**: Tiny evaluator を流用（`BroadcastInDim`/`Transpose`/`Reduce` を実装）
* **精度**: `rtol=1e-3, atol=1e-3`
* **測定ログ**: FLOPs/Bytes/occ/estimate/actual/plan を継続記録

## 6.12 他層との整合（差分）

* **IndexBook**: 連番軸・AxisMap と整合。`layout` を追加
* **Poly‑View**: 解析専用。RB‑SSA にパターン語は持ち込まない
* **Plan**: `tensorize` は **Reduce 名**を参照。`layout_hints` と `Transpose` の整合を明記
* **GPU IR**: 固定骨格に Plan 注入。`Select` は predication へ。**非同期トークン**で依存管理

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

# 8. Schedule Plan（JSON 定義と意味：**非同期拡張**を含む）

## 8.1 フィールドとセマンティクス

* `tile=[BM,BN,BK]` … CTA タイル
* `stages ∈ {2,3}` … SMEM パイプライン段数
* `bind` … 軸の CTA/warp/lane への割付（例：`m.o→block.y`）
* `warp_tile` … warp 内 MMA 形状（例：`64x64`）
* `cache` … どのテンソルをどこでキャッシュ（`smem`）し、どの軸で入替（`at`）
* `vectorize` … グローバル ld/st のベクトル幅
* `predicate_tail` … predication 対象軸
* `epilogue` … acc 直接適用オペ（bias/activation）
* `arch ∈ {"sm80","sm90"}`
* `layout_hints` … スウィズルや出力ストライド指示
* `algo_choice` … `matmul|conv|attention` の実装分岐
* `local_edges` … 値間前渡し（`reg`）や **SMEM リング**（`smem_ring`, depth）
* **新規（非同期）**

  * `async.enable` … `true|false`（TMA/cp.async と Epilogue の重ねを許可）
  * `async.prefetch_depth` … 先読み深さ（例：2）
  * `async.epilogue_overlap` … `true|false`（Epilogue と次タイルロードの並列）
  * `barrier_model` … `"cp_async_group"` or `"mbarrier"`（SM80/SM90）

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
  "local_edges":[{"from":"sum_k","to":"relu","buffer":"reg"}],
  "async":{"enable":true,"prefetch_depth":2,"epilogue_overlap":true},
  "barrier_model":"cp_async_group"
}
```

---

# 9. GPU方言IR（単一MMAテンプレート + **非同期トークン**）

## 9.1 固定骨格と意味

* **SM80**: `cp.async → commit/wait → ldmatrix → mma.sync → epilogue → st.global.vN`
* **SM90**: `tma.load → mbarrier.arrive/wait → wgmma.mma_async → epilogue → st.global.vN`
* **tail** は Plan の `predicate_tail` に従い predicated
* **Epilogue** は **acc** に直適用（GMEM 往復禁止）

## 9.2 **非同期トークンモデル（新規）**

* 目的: TMA/cp.async による**次タイルロード**と **Epilogue** の**重ね**を表現
* 抽象トークン種別

  * `TokLoad` … グローバル→SMEM の着荷完了を表す
  * `TokCompute` … MMA 完了を表す
  * `TokEpilogue` … Epilogue 完了を表す
* ルール

  * `TokLoad` は `Wait(TokLoad)` で消費。SM80 では `commit_group/wait_group`、SM90 では `mbarrier.try_wait.parity` に対応
  * `TokCompute` は次ステップの Epilogue 依存
  * `TokEpilogue` と次タイル `TokLoad` は独立可（`epilogue_overlap:true` のとき）

## 9.3 代表ダンプ（SM90）

```json
{
  "gpu_ir": {
    "arch":"sm90",
    "stmts":[
      {"TmaLoad":{"dst_smem":"smA[s0]","desc":"A_desc","coords":"(m,k0)","mbarrier":"mb","out_token":"tA0"}},
      {"TmaLoad":{"dst_smem":"smB[s0]","desc":"B_desc","coords":"(k0,n)","mbarrier":"mb","out_token":"tB0"}},
      {"MBarrierArrive":{"mbarrier":"mb"}},
      {"Wait":{"token":["tA0","tB0"]}},
      {"Wgmma":{"acc":"acc","a_desc":"smA_tile","b_desc":"smB_tile","shape":"64x128x64","dtype":"fp16_fp32acc","out_token":"tMMA0"}},
      {"Epilogue":{"acc":"acc","ops":["bias"],"in_token":"tMMA0","out_token":"tE0","async":true}},
      {"TmaLoad":{"dst_smem":"smA[s1]","desc":"A_desc","coords":"(m,k1)","mbarrier":"mb","out_token":"tA1"}},
      {"TmaLoad":{"dst_smem":"smB[s1]","desc":"B_desc","coords":"(k1,n)","mbarrier":"mb","out_token":"tB1"}},
      {"StGlobalVec":{"ptr":"C+coff","regs":"acc_vec","pred":"p_tail","in_token":"tE0"}}
    ]
  }
}
```

---

# 10. 変換パイプライン（Front→Tiny→IndexBook→Poly‑View→Region→Plan→GPU）

1. **Frontend→Tiny**
   高位演算を最小語彙に正規化。`Conv2D` は `pad + movement(affine) + mul + reduce`。`Attention` は 2 回の `mul+reduce` と行 softmax
2. **Tiny→IndexBook**
   値ごとに **Axis/Domain/AxisMap** を導出。`reshape/view` は**線形化→再分解**。`%` は**div+ガード**
3. **ContractionPattern 検出**
   `Binary(mul)` 直上の `Reduce(sum)` を候補化。入力が Movement 以外を含まないこと。共通軸／縮約軸を同定
4. **IndexBook→Poly‑View**
   SCoP のみ抽出し、アクセスを `pw_multi_aff` に落とす
5. **Region 生成**
   `can_compute_at` と**ハロー量**で融合可否を判定。**出力のみ MemRef**、中間は値SSAのまま
   **C‑CON1** を適用し、縮約の `body` は **直接 `Read`** による素朴形へ正規化
6. **Plan 付与**
   解析ヒント（並列軸／最小バッファ／バリア）を利用し探索空間を 10〜20 点に絞る。**async** 設定を含む
7. **GPU IR 生成**
   Plan をテンプレに注入。アーキ分岐と tail predication、**非同期トークン**を実装

---

# 11. コストモデルとチューニング

* **基本式**
  `time_est = max( FLOPs/(peak_TF*occ*pipeline_eff),  Bytes/(peak_BW*occ) )`
* **融合の差分評価**
  `Δtime = max(ΔFLOPs/(peak_TF*occ*pipeline_eff), ΔBytes/(peak_BW*occ)) + Δocc_time + penalties − saved_bytes/peak_BW`

  * `penalties = fanout_penalty + layout_penalty(TMA/WGMMA, vectorize) + piecewise_penalty`
* **レイアウトの扱い（新規）**

  * `layout_penalty` は **物理レイアウト変換の必要性**と **TMA/ldmatrix.trans の達成度**で見積もり
  * `Transpose` が論理ビューで済む場合はペナルティ軽微、物質化が必要な場合は高ペナルティ
* **非同期の効果**

  * `async` 有効時は `overlap_gain ≈ min(T_load, T_epilogue)` を差し引き
* **探索空間（MVP）**
  `tile(3–4) × stages{2,3} × warp_tile{2} × vec{4,8,16} × algo_choice（少数） × async{on/off}`
* **STOP 条件**

  * `smem_per_CTA` 推定が上限の **\~80%** 超
  * `regs_per_thr` 増で **CTA/SM=1** に低下
  * `vectorize.width` が **16→8/4** に縮小
  * **SM90**: TMA/WGMMA 要求タイルから外れる

---

# 12. ランタイム／ABI／ShapeSets

* **Graph Signature** を ABI として採用（I/O 役割・可変性・アライン・**layout**）
* **ConstPool**: weights の連続配置と整列
* **Memory Arena**: 生存区間解析に基づく再利用
* **ShapeSets**: 動的形状をクラスタ化し Plan/実測をキャッシュ
* **CUDA Graph**: 形状クラスタごとに 1 個キャプチャし再利用
* **RNG（学習）**: counter‑based（Philox系）で副作用制御
* **エラー**: SMEM/REG 超過・非アフィン・アライン/レイアウト不整合は**明示診断**

---

# 13. デバッグ／検証

* **ダンプ**: `--dump=frontend,tiny,indexbook,poly_view,region,plan,gpu,cu`
* **CPU プレイバック**: Region 内算術を JIT/解釈で実行し数値検証
* **精度検証**: FP32 参照と `rtol=1e-3, atol=1e-3` で比較
* **測定ログ**: `FLOPs/Bytes/occ/estimate/actual/plan` を CSV 追記

---

# 14. 失敗条件と診断（**行動可能な助言つき**）

各診断は `code`, `kind`, `at`, `why`, `suggestion` を含みます。

### 代表診断ダンプ

```json
{
  "diagnostics":[
    {
      "code":"E1001",
      "kind":"BroadcastMismatch",
      "at_op":"Binary(mul)",
      "lhs_shape":["M","K","N"],
      "rhs_shape":["M","K","N'"],
      "why":"右端揃え比較で size>1 の軸 N と N' が不一致です。",
      "suggestion":"BroadcastInDim の自動挿入ができません。片側を expand して次元を一致させるか、計算順序を見直してください。"
    },
    {
      "code":"E1102",
      "kind":"NegativeStrideUnsupported",
      "at_op":"Movement(view)",
      "why":"負ストライド view は非対応です。",
      "suggestion":"代替として Transpose + slice の合成で同等のビューを表現してください。"
    },
    {
      "code":"E1203",
      "kind":"MissingDivGuard",
      "at":"Poly-View",
      "why":"`%` を保持したままです。",
      "suggestion":"`reshape/view` を線形化→再分解し、`div` とガードに展開してください。"
    },
    {
      "code":"W2101",
      "kind":"ExcessivePiecewise",
      "at_value":"Xv",
      "why":"piecewise の union が過大です。",
      "suggestion":"`gist/coalesce/remove_redundancies` を強制し、pad/slice の連鎖を簡約してください。"
    },
    {
      "code":"E1304",
      "kind":"AxisAlignmentMismatch",
      "at":"IndexBook.align_params",
      "why":"連番軸 i2 のサイズが署名と不一致です。",
      "suggestion":"Graph Signature のシンボル名・サイズを再確認し、Tensor の view を修正してください。"
    }
  ]
}
```

---

# 15. 端から端までの例（抜粋）

## 15.1 GEMM + Bias + ReLU

* Frontend 署名＋DAG（§2.2）
* Tiny（§3）
* IndexBook（§4.5）
* Poly‑View（§5.2）
* Region（§6.8.1; **C‑CON1** 準拠）
* Plan（§8.1 例；`async` 有効）
* GPU IR（§9.3；非同期トークン）

## 15.2 Conv 3×3 + SiLU

* Tiny: `pad + movement(affine) + mul + reduce`
* IndexBook: **piecewise** と **ハロー ±1**
* Poly‑View: `pattern="conv"`（`affine_offsets` に `(h+kh-1, w+kw-1)`）
* Region: `local_edges=smem_ring`
* Plan: `stages=3, warp_tile=64x32, vec=8, async.enable=true`
* GPU IR: SM80 骨格

## 15.3 Attention（causal）

* Tiny: QKᵀ→softmax→V（2 contraction + reduce/ewise）
* IndexBook: mask の多様ブロードキャストを 0 写像で正規化
* Poly‑View: `qk_matmul`／`row_softmax`／`pv_matmul`
* Region: `P` を `materialize:"deferred"`、2 Region 連結
* Plan: `"attention":"streaming_softmax_2pass"`（SM90 推奨, `async` 有効）
* GPU IR: `tma + mbarrier + wgmma`、tail は行/列で predicated、Epilogue とロードの重ね


