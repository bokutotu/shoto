近藤さま

ご要望どおり、**tinygrad の ref‑compat（明示ブロードキャスト）**の意味を一切省略せずに維持したまま、**各 lowering 段**で「何を・どう計算しているか」を**アルゴリズム視点**で説明します。対象は `X:[M,K] · W:[K,N] (+ bias:[N]) → Y:[M,N]`、**FP16 入力 / FP32 蓄積 / bias+ReLU 融合**です。

---

## 0. 前提：tinygrad の `dot/matmul` がやっていること（参照仕様）

* `axis_w = -min(w.ndim, 2)` を決める（1D なら -1、2D 以上なら -2）。
* **reshape で 1 次元を 0〜1 個挿入**して、要素積に必要な右端揃えブロードキャストを**明示**で誘発。
* 必要なら **transpose(-1, axis\_w)** で「縮約軸 K を最後尾」へ移す。
* `MUL`（要素積）→ `sum(-1)`（最後の軸 K を縮約）。
* `+ bias` は **bias を \[1,N] に reshape** → `ADD`（右端揃えブロードキャスト）→ `RELU` →（必要なら）`CAST`。

この評価順を、**IR 上で可視**のまま下層へ渡します。

---

# 1. Tiny IR → IndexBook

### 1.1 入出力

* **入力**：tinygrad 互換 UOp DAG
  例（2D×2D）：`VIEW(X), VIEW(W), RESHAPE(X→[M,1,K]), RESHAPE(W→[1,K,N]), PERMUTE(W→[1,N,K]), MUL, REDUCE(sum over -1)`
* **出力**：各値 `v` ごとの

  * 軸集合 `axes = [{id, name, size, kind∈{iter,broadcast,reduce}}]`
  * アクセス式 `AxisMap`（**出力軸→入力軸**のベクトル：アフィン + `floordiv` のみ）
  * 定義域 `Domain`（不等式の集合。必要に応じ piecewise）

### 1.2 全体フロー（擬似コード）

```python
def lower_tiny_to_indexbook(uops, signature):
    topo = topologically_sorted(uops)
    book = {}
    for node in topo:
        if node.uop == "VIEW":
            book[node.out] = leaf_from_signature(node.src)            # axes: (iter), inputs: identity
        elif node.uop == "RESHAPE":
            book[node.out] = reshape_update(book[node.src], node.arg)
        elif node.uop == "PERMUTE":
            book[node.out] = permute_update(book[node.src], node.arg.dims)
        elif node.uop == "MUL":
            book[node.out] = binary_right_align_broadcast(book, node.srcL, node.srcR)
        elif node.uop == "REDUCE":
            book[node.out] = reduce_update(book[node.src], axes=node.arg.axes, acc_dtype=node.arg.acc_dtype)
        elif node.uop in {"ADD","RELU","CAST"}:
            book[node.out] = ewise_update(book, node)                 # 型検査と右端揃えのみ
        else:
            error("unsupported")
    validate(book)
    return book
```

### 1.3 各オペの具体アルゴリズム

**(A) VIEW（葉）**

* 署名の形状をそのまま **iter 軸**として採番。例：`X:[M,K] ⇒ axes=[(m:iter),(k:iter)]`
* `inputs=[]`（自身が元）

**(B) RESHAPE（線形化 → 再分解）**
一般形は

* 線形化：`L = Σ_i (oi * stride_i)`（旧出力軸 `oi`）
* 再分解：新軸 `nj` を `nj = floor(L / stride'_j) % size'_j`
* `%` 起源の条件は Domain に **ガード**として追加（SCoP 維持）
  ただし今回の reshape は **サイズ 1 次元の挿入**のみ：
* 新軸は `kind=broadcast, size=1` として **AxisMap へ定数 0** を入れる。Domain 追加は不要。

**(C) PERMUTE（置換）**

* AxisMap のベクトルを `dims` で置換。

**(D) MUL（右端揃えブロードキャストの合一）**

* 左右の \*\*「有効軸列」\*\*を右端揃えで突き合わせ、次を繰り返す：

  * どちらかが `size=1`（あるいは `kind=broadcast`）なら **出力軸を相手の軸で採用**し、broadcast 側は **AxisMap へ定数 0**。
  * そうでなく両方 `>1` でサイズ一致なら、その軸を採用（両者 AxisMap は恒等）。
  * それ以外は **BroadcastMismatch**（エラー）。
* 出力の `axes` は上記で得た列。`inputs=[{idL,mapL},{idR,mapR}]`

**(E) REDUCE（軸削除）**

* 指定軸（ここでは最後尾 `k`）を **`kind=reduce` とマーク**し **出力から削除**。
* `reduce_axes` に削除前の軸 ID を記録。`acc_dtype` を付与。

**(F) ADD/RELU/CAST（要素演算）**

* 型検査。必要なら **内部で右端揃えブロードキャスト**（`MUL` と同じ手順で 0 写像導入）。

### 1.4 複雑度

* UOp ノード数を `|V|`、各ノードの rank（軸数）の最大を `R` とすると O(|V|·R)。

---

# 2. IndexBook → Poly‑View（解析専用）

### 2.1 目的

* 実行はしない。**SCoP の Domain** と \*\*アクセス式（区分アフィン）\*\*だけを抽出し、依存・並列・パターン検出を容易にする。

### 2.2 アルゴリズム

1. **空間正規化**：IndexBook の **連番軸名**を踏襲（rename しない）。
2. **式変換**：AxisMap を isl 等の `pw_multi_aff` に落とす。`floordiv` は `div` として導入、`%` 由来は Domain 側のガードに。
3. **簡約**：`coalesce/gist/remove_redundancies` を適用し Domain を縮約。
4. **パターン検出（必要なら）**：`Binary(mul)` の上に `Reduce(sum)` が乗る形を探索し、**ContractionPattern=matmul** としてメタ情報に記録（以降の schedule 制約に使う）。

### 2.3 出力

* Domain: `{[m,n,k]: 0≤m<M ∧ 0≤n<N ∧ 0≤k<K}`
* Access: `X[m,k]`, `W[k,n]`, `P[m,n,k]`, `Y[m,n]`
* reduce 軸：`k`、並列軸：`m,n`

---

# 3. Poly‑View → Region Buffer SSA（RB‑SSA）

### 3.1 目的

* **非物質化の最小語彙**（`Read/reshape/transpose/BroadcastInDim/mul/reduce/...`）で、**実行可能な SSA**に落とす。
* 今回は tinygrad 互換（**先に形を揃えてから ewise→reduce**）を維持。

### 3.2 生成手順（擬似コード）

```python
def build_region_from_indexbook(book):
    # 1) iters の決定（出力の自由軸）
    iters = ["m","n"]

    # 2) Movement の正規化（非物質化）
    x1 = reshape(read("X", map=["m","k"]), result_shape=["M",1,"K"])
    w1 = reshape(read("W", map=["k","n"]), result_shape=[1,"K","N"])
    w2 = transpose(w1, permutation=[0,2,1])  # 1,N,K

    # 3) ブロードキャストの明示
    xB = BroadcastInDim(x1, result_shape=["M","N","K"], broadcast_dimensions=[0,2])
    wB = BroadcastInDim(w2, result_shape=["M","N","K"], broadcast_dimensions=[1,2])

    # 4) 要素積→縮約
    p   = mul(xB, wB)
    y_a = reduce_sum(p, axes=["k"], init=0.0f, dtype=fp32)

    # 5) epilogue（bias + relu）
    b1  = reshape(read("bias", map=["n"]), result_shape=[1,"N"])
    y_b = add(y_a, BroadcastInDim(b1, ["M","N"], [1]))
    y_r = relu(y_b)

    yield FP16(y_r) → "Y"
```

### 3.3 妥当性チェック

* `BroadcastInDim` の `result_shape` が両辺一致しているか。
* `reduce` の `axes` が IndexBook の `reduce_axes` と一致するか。
* `acc_dtype`（FP32）が付与されているか。

---

# 4. Region → Poly‑Core（依存・並列・資源見積り）

### 4.1 目的（実行前解析）

* 並列可能軸：`m,n`。縮約軸：`k`。
* tail 軸抽出：`m,n,k`（全軸がタイル端でガード要）。
* **最小バッファ**：`reg`（acc）、`smem`（X と W のリング：深さ 2）（MVP は同期ロードでも OK）。
* バリアヒント：同期版なら `__syncthreads()` で十分（非同期は `cp.async/mbarrier`）。

### 4.2 アルゴリズム（抜粋）

* 依存グラフは単純（`p` は `xB,wB` に依存、`y_a` は `p` に依存…）。**True/Anti/Output** 全て無し（SSA）。
* ハロー（conv で効くが gemm では 0）。
* SMEM 見積り：`BM*BK*sizeof(half) + BK*BN*sizeof(half)` が上限以内かを確認。

---

# 5. Poly‑Core → Schedule Plan（タイル・割付・パイプライン）

### 5.1 入出力

* **入力**：解析結果（並列軸, reduce 軸, バッファ見積もり）
* **出力**：`tile=[BM,BN,BK]`, `bind`, `stages`, `predicate_tail`, `epilogue`, `async` など

### 5.2 タイル探索（MVP ヒューリスティクス）

1. 候補集合：`BM,BN ∈ {64,128}`, `BK ∈ {16,32,64}`, `stages∈{2,3}`。
2. **SMEM 制約**：`BM*BK + BK*BN` が `smem_per_CTA` の \~80% 未満。
3. **Occupancy**：推定レジスタ数 × block サイズで CTA/SM ≥ 2 を優先。
4. **帯域/計算バランス**：`time_est = max( FLOPs/TF, Bytes/BW )` が最小。
   FLOPs ≈ `2*M*N*K`、Bytes ≈ `sizeof(half)*(BM*BK + BK*BN + BM*BN)`（再利用込み）。
5. もっとも良い候補を採用。ここでは **\[64,64,32], stages=2**。

### 5.3 マッピング

* `bind`: `m.o → block.y`, `n.o → block.x`。block 内は（例）`block=(16,16)`, **thread あたり 2×2 のマイクロタイル**。
* `predicate_tail`: `m,n,k` を列挙（ガード生成に使う）。

---

# 6. Schedule Plan → GPU IR（テンプレート注入）

### 6.1 同期版（MVP）の骨格

```
AllocShared As[BM×BK], Bs[BK×BN]
for block_m in 0..ceil(M/BM)-1:
  for block_n in 0..ceil(N/BN)-1:
    acc[thread,2×2] = 0
    for k0 in 0..ceil(K/BK)-1:
      # predicated global→shared
      load_tile As <- X[block_m*BM : , k0*BK : ]
      load_tile Bs <- W[k0*BK : , block_n*BN : ]
      __syncthreads()

      for kk in 0..BK-1:
        # BROADCAST の物理化：
        #   As[* , kk] を N方向に、Bs[kk , *] を M方向に共有
        acc += As[row,kk] * Bs[kk,col]

      __syncthreads()
    epilogue(acc, bias[col])  # add + relu（レジスタ上）
    predicated store Y
```

### 6.2 「明示ブロードキャスト」をどう実現しているか

* `xB:[M,N,K]` は **As の各行（M）と kk（K）を固定すると** **N 全体**に同じ値を使う ⇒ **N 方向の broadcast**。
* `wB:[M,N,K]` は **Bs の各列（N）と kk（K）を固定すると** **M 全体**に同じ値を使う ⇒ **M 方向の broadcast**。
* 物理的には **共有メモリに 1 度だけロード**し、CTA 内の全スレッドが再利用することで broadcast を実現。

---

# 7. GPU IR → CUDA（テキスト生成）

### 7.1 カーネル構造の生成アルゴリズム

1. **シグネチャ**：`(X,W,bias,Y,M,N,K)` と `template<BM,BN,BK>` を出力。
2. **共有メモリ宣言**：`__shared__ half As[BM*BK], Bs[BK*BN];`
3. **スレッド配置**：`block=(16,16)` を前提に `micro_m=micro_n=2` を決める。
4. **タイルループ生成**：

   * **Load ループ**（coalescing）：`t = ty*bdx + tx; for idx=t; idx<tile_elems; idx+=bdx*bdy` でタイルを協調ロード。

     * **OOB は 0 をロード**（tail predication）。
   * **計算ループ**：`for kk in 0..BK-1` で 2×2 のレジスタ `acc` に積和。
5. **Epilogue**：`v = acc + bias[col]; v = max(v,0); Y[...] = fp32_to_fp16(v)` を predicated store。
6. **ホスト側**：grid/block 設定、メモリ確保・転送、リファレンス計算、差分チェック。

### 7.2 coalescing（連続アクセス）の設計

* グローバル→共有のタイルロードは `idx` の一次元イテレーションに展開し、`thread` を線形 ID で割り当てることで **連続要素を別スレッドが順に読む**。
* 行列が行メジャ（row-major）なので、`As` は `i` が連続、`Bs` は `j` が連続になるよう `idx` の展開式を選ぶ。

### 7.3 tail predication の生成

* **ロード**：`(gRow<M && gK<K)` / `(gK<K && gCol<N)` を判定。偽なら `0` を共有メモリへ。
* **ストア**：`(gRow<M && gCol<N)` のときだけ Y へ書き戻す。

---

# 8. Epilogue 融合のアルゴリズム

1. **依存性判定**：`bias[n]` は `m` に依存しない（`Depends(bias, m)=false`）ので **post‑ewise**（R3 規則）として `reduce` の外に安全に置ける。
2. **型処理**：`reduce(dtype=fp32)` の結果を **FP32 のまま** bias 加算・ReLU。最後に `cast(fp16)`。
3. **計算配置**：**レジスタ `acc` に直接適用**。グローバルへ一度も戻さない（GMEM 往復ゼロ）。

---

# 9. 代表的な診断が出るときの検出法

* **BroadcastMismatch**：`right_align_broadcast` 中に「両辺>1 かつ size 不一致」を見つけたら即エラー。
* **AxisAlignmentMismatch**：IndexBook で署名のシンボル（M,N,K）と値の `size` が合わない場合にエラー。
* **AccDtypeMissing**：`REDUCE` に `dtype` が無い場合にエラー（FP16 蓄積は避けたい）。

（いずれも **UOp ごとのローカル検査**＋**最終検証**の二段構えで報告します。）

---

# 10. 計算量と資源見積りの式

* **FLOPs**：`2*M*N*K`（FMA を 2 FLOPs と数える慣習）
* **DRAM Bytes（理想再利用時）**：`sizeof(half)*(M*K + K*N + M*N)`
  タイル再利用の効果は `K/BK` 回のロードで抑えられる。
* **時間見積り**：`T ≈ max( FLOPs / (peak_TF·occ·pipeline_eff), Bytes / (peak_BW·occ) )`
  MVP では `pipeline_eff ≈ 1`（非同期なし）。

---

# 11. 1D×2D / 2D×1D でも同じか？（一般化）

* `1D×2D`（`x:[K] · W:[K,N] → y:[N]`）：
  `x→[K]`、`W→[K,N]→permute→[N,K]`、`MUL→[N,K]`、`sum(-1)`。
  CUDA では **As が 1 行**、**Bs が BN 列**のタイルになり、本質は同じ（broadcast は M 方向が消えるだけ）。
* `2D×1D`（`X:[M,K] · w:[K] → y:[M]`）：
  `X→[M,K]`、`w→[K]`、`MUL→[M,K]`、`sum(-1)`。
  **Bs が 1 列**になるイメージ。

---

# 12. （発展）非同期ロード / WMMA へ拡張する場合のアルゴリズム差分

* **cp.async（SM80） / TMA（SM90）**

  * 先読み深さ `prefetch_depth∈{2,3}` を設定。
  * **トークン**（cp.async group id / mbarrier parity）を IR に持たせ、`Wait(token)` と `Epilogue` の重ねを許す。
* **WMMA/WGMMA**

  * `BK` をハード要求（16 or 32）に合わせ、`warp_tile` を `16×16`（WMMA）や `64×128`（WGMMA）に調整。
  * `BroadcastInDim` / `Transpose` は **`ldmatrix(.trans)` / TMA descriptor** に吸収（論理は不変）。

---

## まとめ

* **各 lowering**は、
  ① tinygrad の **reshape→transpose→明示ブロードキャスト→MUL→sum** という**参照手順を保持**して IndexBook で厳密化し、
  ② RB‑SSA で **非物質化 Movement と BroadcastInDim を明示**、
  ③ Schedule で **タイルと割付**を決め、
  ④ GPU IR テンプレートに注入して、
  ⑤ CUDA では **共有メモリ再利用で broadcast を物理化**、**epilogue をレジスタで融合**、**tail を predication**で処理します。
* どの段でも **“何を（意味）／どこに（軸と領域）／どうやって（実装）”** の分離を保ち、規則は**機械的**に適用できます。

必要でしたら、このアルゴリズムをそのまま動く \*\*最小実装（Python で IR→IndexBook→RB‑SSA ダンプ）\*\*に落とした雛形も用意します。

