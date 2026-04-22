# CUDA Shared Memory Optimization - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现带 shared memory 优化的矩阵乘法内核,并对比 naive 版本的性能

**Architecture:** 在 matrix_mult.cu 中添加两个 CUDA kernel: naive 版本(每个线程计算一个元素) 和 tiled 版本(使用 shared memory 分块)。通过 benchmark 函数对比两者性能。

**Tech Stack:** CUDA C++, Google Benchmark (如项目中有) 或手动计时

---

## Task 1: 添加 Naive 矩阵乘法 Kernel

**Files:**
- Modify: `src/matrix_mult.cu`
- Modify: `include/matrix_mult.h`

- [ ] **Step 1: 在 matrix_mult.h 中添加 naive 函数声明**

在 `include/matrix_mult.h` 末尾添加:

```cpp
// Naive 矩阵乘法 (不使用 cuBLAS,每个线程计算一个输出元素)
template <typename T>
void multiplyMatricesNaive(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB);

extern template void multiplyMatricesNaive<float>(const float*, const float*, float*, int, int, int);
extern template void multiplyMatricesNaive<double>(const double*, const double*, double*, int, int, int);
```

- [ ] **Step 2: 在 matrix_mult.cu 中添加 naive kernel 和 host 函数**

在 `src/matrix_mult.cu` 中,在文件末尾(在 explicit instantiation 之前)添加:

```cpp
namespace cuda_kernel {

// Naive Kernel: 每个线程计算 C 中的一个元素
// C[row, col] = sum(A[row, k] * B[k, col]) for all k
template <typename T>
__global__ void matrixMulNaiveKernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        T sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

}  // namespace cuda_kernel

// Naive 矩阵乘法 Host 函数
template <typename T>
void multiplyMatricesNaive(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB) {
    size_t byteSizeA = numRowsA * numColsA * sizeof(T);
    size_t byteSizeB = numColsA * numColsB * sizeof(T);
    size_t byteSizeC = numRowsA * numColsB * sizeof(T);

    T *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    CUDA_CHECK(cudaMalloc(&deviceMatrixA, byteSizeA));
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, byteSizeB));
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, byteSizeC));

    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, byteSizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, byteSizeB, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numColsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numRowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuda_kernel::matrixMulNaiveKernel<<<numBlocks, threadsPerBlock>>>(
        deviceMatrixA, deviceMatrixB, deviceResultMatrix, numRowsA, numColsA, numColsB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, byteSizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(deviceMatrixA));
    CUDA_CHECK(cudaFree(deviceMatrixB));
    CUDA_CHECK(cudaFree(deviceResultMatrix));
}

template void multiplyMatricesNaive<float>(const float*, const float*, float*, int, int, int);
template void multiplyMatricesNaive<double>(const double*, const double*, double*, int, int, int);
```

- [ ] **Step 3: 编译验证**

Run: `cd /home/mystvio/repos/cu && make build`
Expected: 编译成功,无 warnings

- [ ] **Step 4: Commit**

```bash
git add include/matrix_mult.h src/matrix_mult.cu
git commit -m "feat: add naive matrix multiplication kernel"
```

---

## Task 2: 添加 Tiled 矩阵乘法 Kernel (带 Shared Memory 优化)

**Files:**
- Modify: `include/matrix_mult.h`
- Modify: `src/matrix_mult.cu`

- [ ] **Step 1: 在 matrix_mult.h 中添加 tiled 函数声明**

添加:

```cpp
// Tiled 矩阵乘法 (使用 shared memory 优化)
template <typename T>
void multiplyMatricesTiled(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB);

extern template void multiplyMatricesTiled<float>(const float*, const float*, float*, int, int, int);
extern template void multiplyMatricesTiled<double>(const double*, const double*, double*, int, int, int);
```

- [ ] **Step 2: 在 matrix_mult.cu 中添加 tiled kernel**

在 namespace cuda_kernel 中添加:

```cpp
// Tiled Kernel: 使用 shared memory 优化
// 使用 16x16 的 tile, shared memory 使用 17 列 padding 避免 bank conflict
template <typename T>
__global__ void matrixMulTiledKernel(const T* A, const T* B, T* C, int M, int N, int K) {
    // Shared memory: 使用 17 列来避免 bank conflict
    __shared__ T sharedA[16][17];
    __shared__ T sharedB[16][17];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    T sum = 0;

    // 分阶段处理: 每个阶段处理一个 tile
    for (int phase = 0; phase < (N + 15) / 16; phase++) {
        // 加载 A 的 tile 到 shared memory (合并访问)
        if (row < M && (phase * 16 + tx) < N) {
            sharedA[ty][tx] = A[row * N + phase * 16 + tx];
        } else {
            sharedA[ty][tx] = 0;
        }

        // 加载 B 的 tile 到 shared memory (合并访问)
        if ((phase * 16 + ty) < N && col < K) {
            sharedB[ty][tx] = B[(phase * 16 + ty) * K + col];
        } else {
            sharedB[ty][tx] = 0;
        }

        __syncthreads();

        // 计算这个 tile 的贡献
        for (int i = 0; i < 16; i++) {
            sum += sharedA[ty][i] * sharedB[i][tx];
        }

        __syncthreads();
    }

    // 写入结果
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}
```

- [ ] **Step 3: 在 matrix_mult.cu 中添加 tiled host 函数**

添加 host 函数:

```cpp
template <typename T>
void multiplyMatricesTiled(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB) {
    size_t byteSizeA = numRowsA * numColsA * sizeof(T);
    size_t byteSizeB = numColsA * numColsB * sizeof(T);
    size_t byteSizeC = numRowsA * numColsB * sizeof(T);

    T *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    CUDA_CHECK(cudaMalloc(&deviceMatrixA, byteSizeA));
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, byteSizeB));
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, byteSizeC));

    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, byteSizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, byteSizeB, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numColsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numRowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuda_kernel::matrixMulTiledKernel<<<numBlocks, threadsPerBlock>>>(
        deviceMatrixA, deviceMatrixB, deviceResultMatrix, numRowsA, numColsA, numColsB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, byteSizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(deviceMatrixA));
    CUDA_CHECK(cudaFree(deviceMatrixB));
    CUDA_CHECK(cudaFree(deviceResultMatrix));
}

template void multiplyMatricesTiled<float>(const float*, const float*, float*, int, int, int);
template void multiplyMatricesTiled<double>(const double*, const double*, double*, int, int, int);
```

- [ ] **Step 4: 编译验证**

Run: `cd /home/mystvio/repos/cu && make build`
Expected: 编译成功,无 warnings

- [ ] **Step 5: Commit**

```bash
git add include/matrix_mult.h src/matrix_mult.cu
git commit -m "feat: add tiled matrix multiplication with shared memory optimization"
```

---

## Task 3: 添加 Benchmark 代码

**Files:**
- Modify: `src/main.cpp`
- Modify: `include/matrix_mult.h`

- [ ] **Step 1: 在 matrix_mult.h 中添加 benchmark 函数声明**

添加:

```cpp
// Benchmark 函数: 对比 naive 和 tiled 版本性能
void runMatrixMulBenchmark(int size);
```

- [ ] **Step 2: 在 matrix_mult.cu 中实现 benchmark 函数**

添加:

```cpp
#include <cstdlib>
#include <ctime>

void runMatrixMulBenchmark(int size) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    int M = size, N = size, K = size;

    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C_naive(M * K);
    std::vector<float> h_C_tiled(M * K);
    std::vector<float> h_C_cublas(M * K);

    for (int i = 0; i < M * N; i++) h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
    for (int i = 0; i < N * K; i++) h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;

    cudaEvent_t start, stop;
    float naiveTime, tiledTime, cublasTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\n===== Matrix Multiplication Benchmark (" << size << "x" << size << ") =====" << std::endl;

    cudaEventRecord(start);
    multiplyMatricesNaive(h_A.data(), h_B.data(), h_C_naive.data(), M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naiveTime, start, stop);
    std::cout << "Naive:  " << naiveTime << " ms" << std::endl;

    cudaEventRecord(start);
    multiplyMatricesTiled(h_A.data(), h_B.data(), h_C_tiled.data(), M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiledTime, start, stop);
    std::cout << "Tiled:  " << tiledTime << " ms (speedup: " << naiveTime / tiledTime << "x)" << std::endl;

    cudaEventRecord(start);
    multiplyMatricesOnGPU(h_A.data(), h_B.data(), h_C_cublas.data(), M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublasTime, start, stop);
    std::cout << "cuBLAS: " << cublasTime << " ms (speedup: " << naiveTime / cublasTime << "x)" << std::endl;

    float maxDiff = 0;
    for (int i = 0; i < M * K; i++) {
        float diff = std::abs(h_C_naive[i] - h_C_tiled[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    std::cout << "Max diff (naive vs tiled): " << maxDiff << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

- [ ] **Step 3: 修改 main.cpp 调用 benchmark**

替换 `src/main.cpp`:

```cpp
#include <iostream>
#include "matrix_mult.h"

int main() {
    std::cout << "CUDA Matrix Multiplication Benchmark" << std::endl;

    int sizes[] = {512, 1024, 2048};
    for (int size : sizes) {
        runMatrixMulBenchmark(size);
    }

    return 0;
}
```

- [ ] **Step 4: 编译验证**

Run: `cd /home/mystvio/repos/cu && make build`
Expected: 编译成功

- [ ] **Step 5: 运行测试**

Run: `make run`
Expected: 输出 benchmark 结果,显示 naive/tiled/cuBLAS 的执行时间和加速比

- [ ] **Step 6: Commit**

```bash
git add src/main.cpp src/matrix_mult.cu include/matrix_mult.h
git commit -m "feat: add benchmark for naive vs tiled matrix multiplication"
```

---

## Self-Review Checklist

- [ ] Spec coverage: naive kernel ✓, tiled kernel ✓, shared memory ✓, bank conflict padding ✓, benchmark ✓
- [ ] Placeholder scan: 无 TBD/TODO
- [ ] Type consistency: 所有函数签名一致

---

**Plan complete.** 两个执行选项:

**1. Subagent-Driven (recommended)** - 我dispatch一个subagent逐任务执行,任务间review

**2. Inline Execution** - 在本session中执行任务,带checkpoint

选择哪个?
