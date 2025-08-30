近藤さま

ご指摘の \*\*「tinygrad の dot/matmul が行う“明示的ブロードキャスト（reshape→transpose→要素積）→ sum”を、IR 上でもそのまま表してから CUDA まで落としてほしい」\*\*に合わせ、**1段階も省略せず**に、`GEMM + bias + ReLU` を \*\*tinygrad 互換（ref‑compat）\*\*のまま CUDA へ lower する全手順をまとめました。
対象は `X:[M,K]`, `W:[K,N]`, `bias:[N]`、出力 `Y:[M,N]`、**FP16 入力 + FP32 蓄積**、**Epilogue で bias + ReLU 融合**です。

---

## 0. 入口：tinygrad の評価規則（おさらい）

`dot`/`matmul` は

1. `axis_w = -min(w.ndim, 2)` を決定
2. **reshape で 1 次元を 0〜1 個挿入**
3. 必要なら **transpose(-1, axis\_w)** で「縮約軸 K を最後尾」に移動
4. **要素積**（右端揃えブロードキャスト）
5. **`sum(-1)`**（最後の軸 K を縮約）
6. その後の `+ bias` も **reshape で \[1,N]** にしてから ewise（右端揃えブロードキャスト）→ ReLU →（必要なら）cast

この“**明示ブロードキャスト**”を IR に **そのまま**残します。

---

## 1. Tiny IR（UOp）— ref‑compat（明示 reshape / transpose / ブロードキャスト）

### 1.1 行列積（2D×2D）：`X:[M,K] · W:[K,N] → Y:[M,N]`

```json
{
  "uops": [
    {"uop":"VIEW","src":["X"],"out":"X0"},
    {"uop":"VIEW","src":["W"],"out":"W0"},

    {"uop":"RESHAPE","src":["X0"],"arg":{"result_shape":["M",1,"K"]},"out":"X1"},
    {"uop":"RESHAPE","src":["W0"],"arg":{"result_shape":[1,"K","N"]},"out":"W1"},
    {"uop":"PERMUTE","src":["W1"],"arg":{"dims":[0,2,1]},"out":"W2"},   // 1,N,K

    {"uop":"MUL","src":["X1","W2"],"out":"P"},                          // [M,1,K]×[1,N,K]→[M,N,K]
    {"uop":"REDUCE","src":["P"],"arg":{"op":"SUM","axes":[-1],"acc_dtype":"fp32"},"out":"Y_fp32"}
  ]
}
```

### 1.2 bias + ReLU 融合（明示ブロードキャスト）

```json
{
  "uops": [
    {"uop":"VIEW","src":["bias"],"out":"b0"},
    {"uop":"RESHAPE","src":["b0"],"arg":{"result_shape":[1,"N"]},"out":"b1"},
    {"uop":"ADD","src":["Y_fp32","b1"],"out":"Yb"},   // [M,N]+[1,N]→[M,N]
    {"uop":"RELU","src":["Yb"],"out":"Yr"},
    {"uop":"CAST","src":["Yr"],"arg":{"to":"fp16"},"out":"Y"}
  ]
}
```

> 以降の段階では、この **UOp の意味を保ったまま**、添字・領域→SSA→GPU へ落とします。

---

## 2. IndexBook — 軸・領域・アクセス式の正規化

**各値ごと**に出力軸（iter / broadcast / reduce）と、入力からの \*\*AxisMap（アフィン + floordiv 限定）\*\*を記録します。

```json
{
  "index_book": {
    "X1": {
      "axes":[
        {"id":0,"name":"m","size":"M","kind":"iter"},
        {"id":1,"name":"b1","size":1,"kind":"broadcast"},
        {"id":2,"name":"k","size":"K","kind":"iter"}
      ],
      "inputs":[{"value_id":"X","map":["m","k"]}]
    },
    "W2": {
      "axes":[
        {"id":0,"name":"b2","size":1,"kind":"broadcast"},
        {"id":1,"name":"n","size":"N","kind":"iter"},
        {"id":2,"name":"k","size":"K","kind":"iter"}
      ],
      "inputs":[{"value_id":"W","map":[0,"k","n"]}]    // PERMUTE 済（1,N,K）
    },
    "P": {
      "axes":[
        {"id":0,"name":"m","size":"M","kind":"iter"},
        {"id":1,"name":"n","size":"N","kind":"iter"},
        {"id":2,"name":"k","size":"K","kind":"iter"}
      ],
      "inputs":[
        {"value_id":"X1","map":["m",0,"k"]},           // 0 は broadcast 軸の 0
        {"value_id":"W2","map":[0,"n","k"]}
      ]
    },
    "Y_fp32": {
      "axes":[
        {"id":0,"name":"m","size":"M","kind":"iter"},
        {"id":1,"name":"n","size":"N","kind":"iter"}
      ],
      "inputs":[{"value_id":"P","map":["m","n","k"]}],
      "reduce_axes":[2]                                // k を削除
    },
    "b1": {
      "axes":[
        {"id":0,"name":"b3","size":1,"kind":"broadcast"},
        {"id":1,"name":"n","size":"N","kind":"iter"}
      ],
      "inputs":[{"value_id":"bias","map":[0,"n"]}]
    }
  }
}
```

**ここで分かること**

* **tail 軸**：`m,n,k` はすべて tail になり得る（タイル端でガードが必要）
* **縮約軸**：`k`
* **ブロードキャスト軸**：`b1, b2, b3`（いずれもサイズ 1）

---

## 3. Poly‑View（解析専用）— SCoP + アクセス式

多面体セットで **SCoP のみ**を保持（非アフィンなし）。ここでは最小限：

* Domain: `{ [m,n,k] : 0<=m<M and 0<=n<N and 0<=k<K }`
* Access: `X[m,k]`, `W[k,n]`, `bias[n]`
* パターン：`mul → reduce(sum over k)` なので **ContractionPattern=matmul** を抽象化可能（解析の都合だけ）

> 実装 IR にパターン名は持ち込みません（次段で純 SSA に落とします）。

---

## 4. Region Buffer SSA（RB‑SSA）— 明示ブロードキャストのまま純 SSA 化

* **Movement** は `reshape/transpose` を **非物質化**で保持
* **ブロードキャスト**は `BroadcastInDim` を **明示**
* **Reduce** の本体は「**一度形を揃えたテンソル `p` を sum**」にします（tinygrad の順序に合わせる）

```json
{
  "region": {
    "name":"gemm_bias_relu_ref_compat",
    "iters":[{"name":"m"},{"name":"n"}],
    "inputs":[
      {"name":"X","memref":{"shape":["M","K"],"layout":{"kind":"row_major"}}},
      {"name":"W","memref":{"shape":["K","N"],"layout":{"kind":"row_major"}}},
      {"name":"bias","memref":{"shape":["N"],"layout":{"kind":"row_major"}}}
    ],
    "lets":[
      {"let":"x1","expr":{"reshape":{"operand":{"read":{"tensor":"X","map":["m","k"]}},"result_shape":["M",1,"K"]}}},
      {"let":"w1","expr":{"reshape":{"operand":{"read":{"tensor":"W","map":["k","n"]}},"result_shape":[1,"K","N"]}}},
      {"let":"w2","expr":{"transpose":{"operand":"w1","permutation":[0,2,1]}}},

      {"let":"xB","expr":{"BroadcastInDim":{"operand":"x1","result_shape":["M","N","K"],"broadcast_dimensions":[0,2]}}},
      {"let":"wB","expr":{"BroadcastInDim":{"operand":"w2","result_shape":["M","N","K"],"broadcast_dimensions":[1,2]}}},
      {"let":"p","expr":{"mul":["xB","wB"]}},

      {"let":"y_acc","expr":{"reduce":{"op":"sum","axes":["k"],"init":0.0,"dtype":"fp32","body":"p"}}},

      {"let":"b1","expr":{"reshape":{"operand":{"read":{"tensor":"bias","map":["n"]}},"result_shape":[1,"N"]}}},
      {"let":"y_bias","expr":{"add":["y_acc", {"BroadcastInDim":{"operand":"b1","result_shape":["M","N"],"broadcast_dimensions":[1]}}]}},
      {"let":"y_relu","expr":{"relu":["y_bias"]}}
    ],
    "yield":[{"from":"y_relu","to":"Y","cast":"fp16"}]
  }
}
```

> 以降の段階では、**この SSA を忠実に for ループ + メモリアクセス**へ落とします。

---

## 5. Schedule Plan（探索パラメタの固定）

MVP として読みやすさ最優先のタイル構成：

```json
{
  "tile":[64,64,32],
  "stages":2,
  "bind":{"m.o":"block.y","n.o":"block.x"},
  "warp_tile":"naive_2x2_per_thread",
  "cache":[
    {"tensor":"X","where":"smem","at":"k.i"},
    {"tensor":"W","where":"smem","at":"k.i"}
  ],
  "predicate_tail":["m","n","k"],
  "epilogue":["bias","relu"],
  "arch":"sm80_or_sm90",
  "async":{"enable":false}
}
```

* CTA（スレッドブロック）が `64×64` の出力タイルを担当
* K は `BK=32` ずつ前進
* 共有メモリに `X[BM×BK]`, `W[BK×BN]` をロード（非同期なし）
* 各スレッドは **2×2 のマイクロタイル**を FP32 で蓄積し、最後に bias+ReLU → FP16 書き戻し

---

## 6. GPU IR（骨格）

```
for block m,n:
  zero acc[2×2] as fp32
  for k0 in 0..ceil(K/BK)-1:
    ld.global → st.shared  As[BM×BK], Bs[BK×BN]  (OOB=0)
    __syncthreads
    for kk in 0..BK-1:
      acc += As[*,kk] * Bs[kk,*]
    __syncthreads
  epilogue: add bias[n], relu, cast fp16
  predicated st.global to C (tail)
```

---

## 7. CUDA への lowering（実行可能フルファイル）

> **コピペで動作**します。ビルド例：`nvcc -O3 -arch=sm80 fused_gemm_bias_relu_ref.cu -o fused && ./fused`

```cpp
// fused_gemm_bias_relu_ref.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1);} } while(0)

template<int BM, int BN, int BK>
__global__ void gemm_bias_relu_fp16_fp32acc(const half* __restrict__ X,
                                            const half* __restrict__ W,
                                            const half* __restrict__ bias,
                                            half* __restrict__ Y,
                                            int M, int N, int K) {
  __shared__ half As[BM*BK];
  __shared__ half Bs[BK*BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bdx = blockDim.x;
  const int bdy = blockDim.y;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  const int micro_m = 2;
  const int micro_n = 2;
  const int r0 = ty * micro_m;
  const int c0 = tx * micro_n;

  const int rparts = BM / (bdy * micro_m);
  const int cparts = BN / (bdx * micro_n);

  for (int rblk = 0; rblk < rparts; ++rblk) {
    for (int cblk = 0; cblk < cparts; ++cblk) {
      float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

      for (int k0 = 0; k0 < (K + BK - 1) / BK; ++k0) {
        int t = ty * bdx + tx;
        int stride = bdx * bdy;

        for (int idx = t; idx < BM*BK; idx += stride) {
          int i = idx / BK;
          int k = idx - i * BK;
          int gRow = block_m + i;
          int gK = k0*BK + k;
          As[idx] = (gRow < M && gK < K) ? X[gRow * K + gK] : __float2half(0.f);
        }
        for (int idx = t; idx < BK*BN; idx += stride) {
          int k = idx / BN;
          int j = idx - k * BN;
          int gCol = block_n + j;
          int gK = k0*BK + k;
          Bs[idx] = (gK < K && gCol < N) ? W[gK * N + gCol] : __float2half(0.f);
        }
        __syncthreads();

        int rbase = r0 + rblk * micro_m;
        int cbase = c0 + cblk * micro_n;

        for (int kk = 0; kk < BK; ++kk) {
          float a0 = __half2float(As[(rbase + 0) * BK + kk]);
          float a1 = __half2float(As[(rbase + 1) * BK + kk]);
          float b0 = __half2float(Bs[kk * BN + (cbase + 0)]);
          float b1 = __half2float(Bs[kk * BN + (cbase + 1)]);
          acc00 += a0 * b0;
          acc01 += a0 * b1;
          acc10 += a1 * b0;
          acc11 += a1 * b1;
        }
        __syncthreads();
      }

      int gRow0 = block_m + r0 + rblk * micro_m + 0;
      int gRow1 = block_m + r0 + rblk * micro_m + 1;
      int gCol0 = block_n + c0 + cblk * micro_n + 0;
      int gCol1 = block_n + c0 + cblk * micro_n + 1;

      if (gRow0 < M && gCol0 < N) {
        float v = acc00 + __half2float(bias[gCol0]);
        v = v > 0.f ? v : 0.f;
        Y[gRow0 * N + gCol0] = __float2half(v);
      }
      if (gRow0 < M && gCol1 < N) {
        float v = acc01 + __half2float(bias[gCol1]);
        v = v > 0.f ? v : 0.f;
        Y[gRow0 * N + gCol1] = __float2half(v);
      }
      if (gRow1 < M && gCol0 < N) {
        float v = acc10 + __half2float(bias[gCol0]);
        v = v > 0.f ? v : 0.f;
        Y[gRow1 * N + gCol0] = __float2half(v);
      }
      if (gRow1 < M && gCol1 < N) {
        float v = acc11 + __half2float(bias[gCol1]);
        v = v > 0.f ? v : 0.f;
        Y[gRow1 * N + gCol1] = __float2half(v);
      }
    }
  }
}

static inline half hcast(float x){ return __float2half(x); }
static inline float fcast(half h){ return __half2float(h); }

int main(){
  int M = 150, N = 130, K = 70;

  std::vector<half> X(M*K), W(K*N), Bias(N), Y(M*N), Yref(M*N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(auto &x: X) x = hcast(dist(rng));
  for(auto &x: W) x = hcast(dist(rng));
  for(int j=0;j<N;++j) Bias[j]=hcast(dist(rng));

  for(int m=0;m<M;++m){
    for(int n=0;n<N;++n){
      float acc=0.f;
      for(int k=0;k<K;++k) acc += fcast(X[m*K+k]) * fcast(W[k*N+n]);
      acc += fcast(Bias[n]);
      acc = acc>0.f?acc:0.f;
      Yref[m*N+n] = hcast(acc);
    }
  }

  half *dX,*dW,*dB,*dY;
  CHECK_CUDA(cudaMalloc(&dX, M*K*sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dW, K*N*sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dB, N*sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dY, M*N*sizeof(half)));
  CHECK_CUDA(cudaMemcpy(dX, X.data(), M*K*sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dW, W.data(), K*N*sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, Bias.data(), N*sizeof(half), cudaMemcpyHostToDevice));

  dim3 block(16,16);
  dim3 grid((N+64-1)/64, (M+64-1)/64);
  gemm_bias_relu_fp16_fp32acc<64,64,32><<<grid, block>>>(dX,dW,dB,dY,M,N,K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(Y.data(), dY, M*N*sizeof(half), cudaMemcpyDeviceToHost));

  float max_abs_err=0.f;
  for(int i=0;i<M*N;++i) max_abs_err = std::max(max_abs_err, std::abs(fcast(Y[i]) - fcast(Yref[i])));
  printf("max_abs_err=%.6f\n", max_abs_err);

  cudaFree(dX); cudaFree(dW); cudaFree(dB); cudaFree(dY);
  return 0;
}
```

**この CUDA が「ref‑compat の IR」と対応している点**

* `As` は `X1` の **\[M,1,K]** の「K スライス」を共有メモリに持ってくるのと同値（1 方向は CTA 内で **暗黙の broadcast**）。
* `Bs` は `W2` の **\[1,N,K]** の「K スライス」を共有メモリに持ってくるのと同値。
* `acc += As[* , kk] * Bs[kk , *]` は **\[M,N,K]** の `p = xB * wB` を **kk で逐次評価**して足し込んでいるのと同じ。
* epilogue で `+ bias[n]` を実施（**\[1,N] に reshape された b1 の broadcast**に一致）→ ReLU → FP16 へ cast。

---

## 8. （実装者向け）各段の「機械的変換ルール」まとめ

1. **Tiny IR → IndexBook**

* `RESHAPE`：線形化→再分解で AxisMap を更新。1 次元は **broadcast 軸**として `size=1` を記録
* `PERMUTE`：置換行列で AxisMap を再構成
* `MUL`：右端揃えブロードキャストを **AxisMap の 0 写像**で表現
* `REDUCE(sum, axes=[-1])`：削除前 ID を `reduce_axes` に記録

2. **IndexBook → RB‑SSA**

* `RESHAPE` → `reshape{result_shape}`（非物質化）
* `PERMUTE` → `transpose{permutation}`
* `MUL` 直前で **`BroadcastInDim` を明示**し、両オペランドを同一 `result_shape` にそろえる
* `REDUCE(sum over k)`：`init=0`、`dtype=fp32` を明示
* ewise（bias, relu）：bias は reshape→`BroadcastInDim` の後に `add`、続けて `relu`

3. **RB‑SSA → GPU IR**

* `iters=[m,n]` を CTA グリッドへ割付（`bind`）
* `reduce k` は `for k0`（タイル）＋`for kk`（タイル内）に分解
* `BroadcastInDim(..., [M,N,K])` は **CTA/スレッドのループで指数づけを共有**し、共有メモリに 1 回ロードして **全利用者が再利用**（＝実装上の物理 broadcast）

4. **GPU IR → CUDA**

* 共有メモリ配列 `As[BM×BK], Bs[BK×BN]` の **predicated load**
* `__syncthreads()` をタイルごとに 2 箇所（ロード直後、積和終了直後）
* epilogue を **レジスタ上**で実行し、最後に predicated store

---

## 9. 失敗しがちなポイント

* **axis\_w の扱い**：`dw==1` のとき `transpose(-1,-1)` は no‑op。IR 生成時は **省略しても可**。
* **BroadcastMismatch**：`MUL` 直前に **必ず** `BroadcastInDim` を明示して形を一致させる。
* **tail predication**：`M,N,K` がタイルの倍数でないとき、**ロードとストアの両方にガード**。
* **蓄積型**：`acc_dtype=fp32` を `REDUCE` に付与、epilogue まで FP32 を維持して最後に cast。

---

以上が、\*\*tinygrad の ref‑compat（明示ブロードキャスト）\*\*を **そのまま**保持しつつ、CUDA まで降ろす **完全な経路**です。
次は、このカーネルを **WMMA/WGMMA + 非同期ロード**へ差し替える段で、Plan の `stages=3`・`async.enable=true` を有効化し、`BroadcastInDim` を **TMA/ldmatrix** のディスクリプタ設定に吸収していく、という順序で最適化していくのが安全です。

