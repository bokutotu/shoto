#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <vector>

// 最小限のNVRTCサンプル：単純なカーネルをコンパイルしてPTXを取得

const char* simpleKernel = R"(
extern "C" __global__ void simple_kernel() {
    // 何もしない最小限のカーネル
    return;
}
)";

int main() {
    // 1. NVRTCプログラムの作成
    nvrtcProgram program;
    nvrtcResult result = nvrtcCreateProgram(&program, 
                                           simpleKernel, 
                                           "simple.cu", 
                                           0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) {
        std::cerr << "Failed to create program: " << nvrtcGetErrorString(result) << std::endl;
        return 1;
    }
    
    // 2. コンパイル
    const char* options[] = {"--gpu-architecture=compute_70"};
    result = nvrtcCompileProgram(program, 1, options);
    
    // 3. コンパイルログの確認
    size_t logSize;
    nvrtcGetProgramLogSize(program, &logSize);
    if (logSize > 1) {
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(program, log.data());
        std::cout << "Compilation log:\n" << log.data() << std::endl;
    }
    
    if (result != NVRTC_SUCCESS) {
        std::cerr << "Compilation failed: " << nvrtcGetErrorString(result) << std::endl;
        nvrtcDestroyProgram(&program);
        return 1;
    }
    
    // 4. PTXの取得
    size_t ptxSize;
    nvrtcGetPTXSize(program, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(program, ptx.data());
    
    std::cout << "Successfully compiled! PTX size: " << ptxSize << " bytes" << std::endl;
    std::cout << "\nFirst 200 characters of PTX:\n" << std::string(ptx.begin(), ptx.begin() + 200) << "..." << std::endl;
    
    // 5. クリーンアップ
    nvrtcDestroyProgram(&program);
    
    return 0;
}