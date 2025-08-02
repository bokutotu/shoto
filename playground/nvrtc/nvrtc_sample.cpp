#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <vector>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CUDA Error: " << errStr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define NVRTC_CHECK(call) \
    do { \
        nvrtcResult err = call; \
        if (err != NVRTC_SUCCESS) { \
            std::cerr << "NVRTC Error: " << nvrtcGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// シンプルなベクトル加算カーネル
const char* kernelSource = R"(
extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

int main() {
    // CUDA初期化
    CUDA_CHECK(cuInit(0));
    
    // デバイス取得
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    
    // コンテキスト作成
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));
    
    // NVRTCでプログラムを作成
    nvrtcProgram program;
    NVRTC_CHECK(nvrtcCreateProgram(&program, kernelSource, "vector_add.cu", 0, nullptr, nullptr));
    
    // コンパイルオプション
    const char* options[] = {
        "--gpu-architecture=compute_70",  // GPU アーキテクチャに応じて変更
        "--fmad=false"
    };
    
    // コンパイル
    nvrtcResult compileResult = nvrtcCompileProgram(program, 2, options);
    
    // コンパイルログの取得
    size_t logSize;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program, &logSize));
    if (logSize > 1) {
        std::vector<char> log(logSize);
        NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));
        std::cout << "Compilation log: " << log.data() << std::endl;
    }
    
    if (compileResult != NVRTC_SUCCESS) {
        std::cerr << "Compilation failed!" << std::endl;
        return 1;
    }
    
    // PTXの取得
    size_t ptxSize;
    NVRTC_CHECK(nvrtcGetPTXSize(program, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
    
    // プログラムの破棄
    NVRTC_CHECK(nvrtcDestroyProgram(&program));
    
    // PTXからモジュールをロード
    CUmodule module;
    CUDA_CHECK(cuModuleLoadData(&module, ptx.data()));
    
    // カーネル関数の取得
    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "vector_add"));
    
    // データの準備
    int n = 1024;  // const を削除
    const int bytes = n * sizeof(float);
    
    // ホストメモリ
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);
    
    // デバイスメモリ
    CUdeviceptr d_a, d_b, d_c;
    CUDA_CHECK(cuMemAlloc(&d_a, bytes));
    CUDA_CHECK(cuMemAlloc(&d_b, bytes));
    CUDA_CHECK(cuMemAlloc(&d_c, bytes));
    
    // データをデバイスへコピー
    CUDA_CHECK(cuMemcpyHtoD(d_a, h_a.data(), bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_b, h_b.data(), bytes));
    
    // カーネル起動パラメータ
    void* args[] = { &d_a, &d_b, &d_c, &n };
    
    // カーネル起動
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    CUDA_CHECK(cuLaunchKernel(kernel,
                             blocksPerGrid, 1, 1,    // grid dimensions
                             threadsPerBlock, 1, 1,  // block dimensions
                             0, nullptr,             // shared memory & stream
                             args, nullptr));        // arguments
    
    // 同期
    CUDA_CHECK(cuCtxSynchronize());
    
    // 結果をホストへコピー
    CUDA_CHECK(cuMemcpyDtoH(h_c.data(), d_c, bytes));
    
    // 結果の確認
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            success = false;
            std::cerr << "Error at index " << i << ": expected 3.0, got " << h_c[i] << std::endl;
            break;
        }
    }
    
    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    }
    
    // クリーンアップ
    CUDA_CHECK(cuMemFree(d_a));
    CUDA_CHECK(cuMemFree(d_b));
    CUDA_CHECK(cuMemFree(d_c));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuCtxDestroy(context));
    
    return 0;
}