
# Tiny IR 仕様

> 目的：**NodeId のみ**で値と依存を表し、**Movement + 要素演算 + Reduce**だけで GEMM/Conv/Attention を**ナイーブに**記述できる最小 IR を定義します。
> 非目的：実メモリ移動・同期・レイアウト最適化（これらは後段）。
> 方針：**用語・語彙の揺れを排し**、記述規則と検証条件を明文化します。

---

## 0. 基本原則

* **識別子**: すべてのノードは **`id`（NodeId）**で一意に識別し、**`name/out` は持たない**。
* **語彙**: UOp は **Movement／Unary／Binary／Reduce／Cast／I/O**のみ。
* **ナイーブ表現**: 畳み込み・行列積・アテンションは
  **Movement（転置・ブロードキャスト・ビュー）→ 要素ごとの `MUL` → `REDUCE(SUM)`** で記述する。
* **ブロードキャスト規約**: 右端揃え。二項演算は **同形**に正規化（必要な `EXPAND` を事前挿入）。
* **蓄積精度**: `REDUCE` の `dtype` は **必須**（既定推奨は fp32）。
* **副作用**: なし（純関数 DAG）。外部 I/O は `INPUT` ノードでのみ表す。

---

## 1. データモデル

### 1.1 ノード

```json
{
  "id": "n123",
  "uop": "ADD" | "MUL" | "REDUCE" | "EXPAND" | "PERMUTE" | "RESHAPE" | "PAD" | "SHRINK" | "FLIP" | "VIEW" | "CAST" | "INPUT" | "WHERE" | "MAX" | "MIN" | "SUB" | "FDIV" | "EXP2" | "RELU",
  "src": ["n10", "n11", 0.5],   // NodeId または即値（順序保持）
  "arg": { ... },               // UOp 固有属性（下表）
  "tag": "debug-or-hint"        // 任意。最適化に影響しない
}
```

* **出力は暗黙に「このノードの結果」**。上流は `src` に NodeId で接続する。
* **外部テンソル**は `INPUT` で導入（`tensor_id`, `dtype`, `shape` 必須）。

### 1.2 形状・型

* `shape`: `Int` または記号 `Sym` の配列。
* `dtype`: `fp16 | bf16 | fp32 | i32 | bool`。
* 暗黙キャスト**禁止**（`CAST` を明示）。

---

## 2. 語彙（UOp）— **Movement 統一**

### 2.1 Movement（**非物質化**）

| UOp       | 主な `arg`                                           | 役割                                                                                                     |
| --------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `RESHAPE` | `{"result_shape":[…]}`                             | 総要素数保存の再配置。                                                                                            |
| `PERMUTE` | `{"perm":[…]}`                                     | 次元置換。                                                                                                  |
| `EXPAND`  | `{"result_shape":[…], "broadcast_dimensions":[…]}` | 右端揃えブロードキャスト。                                                                                          |
| `PAD`     | `{"pad":[[lo,hi]...], "value":0}`                  | 域外を定数で埋める。                                                                                             |
| `SHRINK`  | `{"lo":[…], "hi":[…], "step":[…]?}`                | 連続スライス（`step>1` 可）。                                                                                    |
| `FLIP`    | `{"axes":[…]}`                                     | 指定軸の反転（負ストライド）。                                                                                        |
| `VIEW`    | `{"result_shape":[…], "index_map":{...}}`          | **一般アフィン再写像**（`as_strided`/スライディング窓を包含）。 `index_map` は出力軸から入力軸への\*\*アフィン＋`floordiv`\*\*式。 *必要最小限のみ使用*。 |

> 注：ご相談の通り「View という**概念**は Movement の一部」に統一しています。**`VIEW` UOp 自体は必須ではありません**が、Conv 等のスライディング窓を**ナイーブに値複製なし**で表すため、**一般アフィンな再写像**が必要になる場合のみ使用します（代替は Region 以降での gather 化）。

### 2.2 Base（算術・縮約・I/O）

* **Unary**: `NEG, EXP2, RELU, RSQRT, CAST{to}, …`
* **Binary/ternary**: `ADD, SUB, MUL, FDIV, MAX, MIN, WHERE`
* **Reduce**: `REDUCE{op:"SUM|MAX|MIN", axes:[…], dtype:"fp32|…", init?}`
* **I/O**: `INPUT{tensor_id, dtype, shape, storage?}`

---

## 3. 形状整合と型規則

1. **二項演算は同形**：不足次元・サイズ 1 は `EXPAND` で明示的に揃える。
2. **型一致**：二項演算の dtype は一致。異なる場合は事前 `CAST`。
3. **Reduce**：`axes` は出力から**削除**（keepdim=false）。`dtype` は**必須**。

---

## 4. エラー・検証

* `BroadcastMismatch`（右端揃えで両辺>1 不一致）
* `AxisSizeMismatch`（`RESHAPE` で総要素数不一致）
* `InvalidPermutation`（`PERMUTE` の重複/欠落）
* `AccDtypeMissing`（`REDUCE` に `dtype` 無し）
* `ExcessivePiecewise`（`PAD/SHRINK/VIEW` で分割過多）

---

## 5. 代表ナイーブ記述（**`CONTRACT` 不使用**）

### 5.1 GEMM（A\[M,K] × B\[K,N] → C\[M,N]）

```json
{
  "uops": [
    {"id":"n0","uop":"INPUT","arg":{"tensor_id":"A","dtype":"fp16","shape":["M","K"]}},
    {"id":"n1","uop":"INPUT","arg":{"tensor_id":"B","dtype":"fp16","shape":["K","N"]}},

    {"id":"n2","uop":"RESHAPE","src":["n0"],"arg":{"result_shape":["M",1,"K"]}},
    {"id":"n3","uop":"PERMUTE","src":["n1"],"arg":{"perm":[1,0]}},                 // B^T: [N,K]
    {"id":"n4","uop":"RESHAPE","src":["n3"],"arg":{"result_shape":[1,"N","K"]}},   // [1,N,K]

    {"id":"n5","uop":"EXPAND","src":["n2"],"arg":{"result_shape":["M","N","K"],"broadcast_dimensions":[0,2]}},
    {"id":"n6","uop":"EXPAND","src":["n4"],"arg":{"result_shape":["M","N","K"],"broadcast_dimensions":[1,2]}},

    {"id":"n7","uop":"MUL","src":["n5","n6"]},
    {"id":"n8","uop":"REDUCE","src":["n7"],"arg":{"op":"SUM","axes":["K"],"dtype":"fp32"}},
    {"id":"n9","uop":"CAST","src":["n8"],"arg":{"to":"fp16"}}
  ]
}
```

> B の転置（`PERMUTE`）は**必須ではありません**。`RESHAPE`/`EXPAND` の組で同形化できる場合は省略可。

---

### 5.2 Conv2D 3×3（stride=2, pad=1）— **スライディング窓を Movement で記述**

```json
{
  "uops": [
    {"id":"x",  "uop":"INPUT","arg":{"tensor_id":"X","dtype":"fp16","shape":["N","Cin_total","H","W"]}},
    {"id":"w",  "uop":"INPUT","arg":{"tensor_id":"W","dtype":"fp16","shape":["Cout","Cin","Kh","Kw"]}},
    {"id":"b",  "uop":"INPUT","arg":{"tensor_id":"bias","dtype":"fp16","shape":["Cout"]}},  // 無ければ後で未使用

    // X.pad(padding_)._pool(HW, stride, dilation) -> (N, G*Cin, Oy, Ox, Kh, Kw)
    {"id":"xp", "uop":"PAD",   "src":["x"], "arg":{"pad":[[0,0],[0,0],["pt","pb"],["pl","pr"]], "value":0}},
    {"id":"xw", "uop":"POOL", "src":["xp"], "arg":{"kernel":["Kh","Kw"], "stride":["Sh","Sw"], "dilation":["Dh","Dw"]}},

    // x.reshape(N, G, Cin, 1, Oy, Ox, Kh, Kw).expand(..., Rcout, ...).permute(N, G, Rcout, Oy, Ox, Cin, Kh, Kw)
    {"id":"x1", "uop":"RESHAPE","src":["xw"], "arg":{"result_shape":["N","G","Cin",1,"Oy","Ox","Kh","Kw"]}},
    {"id":"x2", "uop":"EXPAND", "src":["x1"], "arg":{"result_shape":["N","G","Cin","Rcout","Oy","Ox","Kh","Kw"]}},
    {"id":"x3", "uop":"PERMUTE","src":["x2"], "arg":{"perm":[0,1,3,4,5,2,6,7]}},

    // weight.reshape(1, G, Rcout, 1, 1, Cin, Kh, Kw) -> expand to (N, G, Rcout, Oy, Ox, Cin, Kh, Kw)
    {"id":"w1", "uop":"RESHAPE","src":["w"],  "arg":{"result_shape":["G","Rcout","Cin","Kh","Kw"]}},
    {"id":"w2", "uop":"RESHAPE","src":["w1"], "arg":{"result_shape":[1,"G","Rcout",1,1,"Cin","Kh","Kw"]}},
    {"id":"w3", "uop":"EXPAND", "src":["w2"], "arg":{"result_shape":["N","G","Rcout","Oy","Ox","Cin","Kh","Kw"]}},

    // 要素積 -> 最後の 3 軸（Cin, Kh, Kw）で sum（keepdim=false）
    {"id":"p",  "uop":"MUL",    "src":["x3","w3"]},
    {"id":"yacc","uop":"REDUCE","src":["p"],  "arg":{"op":"SUM","axes":["Cin","Kh","Kw"],"dtype":"fp32"}},   // (N, G, Rcout, Oy, Ox)

    // 形状を (N, Cout, Oy, Ox) に集約、bias をブロードキャスト加算（あれば）
    {"id":"y0","uop":"RESHAPE","src":["yacc"], "arg":{"result_shape":["N","Cout","Oy","Ox"]}},
    {"id":"b1","uop":"RESHAPE","src":["b"],    "arg":{"result_shape":[1,"Cout",1,1]}},
    {"id":"y1","uop":"ADD",     "src":["y0","b1"]},

    // 必要に応じて出力 dtype を fp16 へ
    {"id":"y", "uop":"CAST",    "src":["y1"],  "arg":{"to":"fp16"}}
  ]
}
```

> ここでの `VIEW` は**一般アフィン再写像**（`as_strided` 相当）として最小限に使用しています。`VIEW` を使わずに書く場合、`SHRINK(step=2)` を複製して 3×3 オフセット分の 9 個のサブテンソルを作り、逐次 `ADD` する方法もありますが、**グラフの肥大化**と **最適化難**が大きいため推奨しません（語彙を増やさない制約下での実用的最小手）。

---

### 5.3 Attention（row‑softmax まで：Q\[B,H,M,D], K\[B,H,N,D]）

```json
{
  "uops":[
    {"id":"n0","uop":"INPUT","arg":{"tensor_id":"Q","dtype":"fp16","shape":["B","H","M","D"]}},
    {"id":"n1","uop":"INPUT","arg":{"tensor_id":"K","dtype":"fp16","shape":["B","H","N","D"]}},

    {"id":"n2","uop":"PERMUTE","src":["n1"],"arg":{"perm":[0,1,3,2]}},                 // K^T: [B,H,D,N]
    {"id":"n3","uop":"RESHAPE","src":["n0"],"arg":{"result_shape":["B","H","M",1,"D"]}},
    {"id":"n4","uop":"RESHAPE","src":["n2"],"arg":{"result_shape":["B","H",1,"N","D"]}},

    {"id":"n5","uop":"EXPAND","src":["n3"],"arg":{"result_shape":["B","H","M","N","D"],"broadcast_dimensions":[0,1,2,4]}},
    {"id":"n6","uop":"EXPAND","src":["n4"],"arg":{"result_shape":["B","H","M","N","D"],"broadcast_dimensions":[0,1,3,4]}},

    {"id":"n7","uop":"MUL","src":["n5","n6"]},
    {"id":"n8","uop":"REDUCE","src":["n7"],"arg":{"op":"SUM","axes":["D"],"dtype":"fp32"}},  // S: [B,H,M,N]

    {"id":"n9","uop":"REDUCE","src":["n8"],"arg":{"op":"MAX","axes":["N"],"dtype":"fp32"}},   // Mrow
    {"id":"n10","uop":"SUB","src":["n8","n9"]},                                               // S0
    {"id":"n11","uop":"MUL","src":["n10",1.442695]},                                          // log2(e)
    {"id":"n12","uop":"EXP2","src":["n11"]},                                                  // A
    {"id":"n13","uop":"REDUCE","src":["n12"],"arg":{"op":"SUM","axes":["N"],"dtype":"fp32"}}, // Z
    {"id":"n14","uop":"FDIV","src":["n12","n13"]},                                            // P (fp32)
    {"id":"n15","uop":"CAST","src":["n14"],"arg":{"to":"fp16"}}
  ]
}
```
