# NVRTC リファレンス実装

shotoプロジェクトでNVRTCを実装する際のリファレンスとなるC++およびHaskellの実装例。

## ファイル構成

### C++実装
- **nvrtc_minimal.cpp** - NVRTCの基本的な使い方（PTX生成まで）
- **nvrtc_sample.cpp** - 完全な例（ベクトル加算の実行まで）

### Haskell実装
- **NvrtcMinimalSimple.hs** - 最小限のFFIバインディング例
- **NvrtcSample.hs** - CUDA Driver APIを含む完全な実装例

## ビルドと実行

### C++版
```bash
# ビルド
make

# 実行
./nvrtc_minimal
./nvrtc_sample
```

### Haskell版
```bash
# コンパイル（重要: stdc++のリンクが必要）
ghc -o program program.hs -lcuda -lnvrtc -lstdc++ -L/usr/local/cuda/lib64

# または runghc で実行
runghc -lcuda -lnvrtc -lstdc++ program.hs
```

## 重要なポイント

### Cabalファイルの設定
```yaml
extra-libraries:  cuda
                , nvrtc
                , stdc++  # NVRTCの依存関係で必須
extra-lib-dirs:   /usr/local/cuda/lib64
```

### FFI宣言の例
```haskell
foreign import ccall "nvrtcCreateProgram" nvrtcCreateProgram :: 
    Ptr NvrtcProgram -> CString -> CString -> CInt -> Ptr CString -> Ptr CString -> IO NvrtcResult
```

## 注意事項
- `libstdc++`のリンクを忘れると、NVRTCの内部で`libnvrtc-builtins.so`のロードに失敗する
- Nix環境では特に注意が必要