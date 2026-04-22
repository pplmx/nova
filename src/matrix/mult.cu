#include <type_traits>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include "cuda/device/device_utils.h"
#include "matrix/mult.h"

// Function to perform matrix multiplication on the GPU using cuBLAS
// This function transfers the input matrices from the host (CPU) to the device (GPU),
// executes the matrix multiplication on the GPU, and retrieves the result back to the host.
// Parameters:
// - hostMatrixA: Pointer to the first matrix (A) on the host (CPU)
// - hostMatrixB: Pointer to the second matrix (B) on the host (CPU)
// - hostResultMatrix: Pointer to the result matrix (C) on the host (CPU)
// - numRowsA: Number of rows in matrix A
// - numColsA: Number of columns in matrix A (and rows in matrix B)
// - numColsB: Number of columns in matrix B
template <typename T>
void multiplyMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix, int numRowsA, int numColsA, int numColsB) {
    // Calculate the size of matrices A, B, and C in bytes
    size_t byteSizeA = numRowsA * numColsA * sizeof(T);
    size_t byteSizeB = numColsA * numColsB * sizeof(T);
    size_t byteSizeC = numRowsA * numColsB * sizeof(T);

    // Device (GPU) memory pointers for matrices A, B, and result matrix C
    T *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    // Allocate memory for matrices on the GPU
    CUDA_CHECK(cudaMalloc(&deviceMatrixA, byteSizeA));       // Allocate memory for matrix A on the GPU
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, byteSizeB));       // Allocate memory for matrix B on the GPU
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, byteSizeC));  // Allocate memory for result matrix C on the GPU

    // Copy matrices A and B from the host (CPU) to the device (GPU)
    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, byteSizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, byteSizeB, cudaMemcpyHostToDevice));

    // Create a cuBLAS handle for matrix multiplication
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Define alpha and beta scalars for the matrix multiplication: C = alpha * A * B + beta * C
    const T alpha = 1.0;
    const T beta = 0.0;

    // Perform matrix multiplication using cuBLAS based on the type of T (float or double)
    // Matrices are stored row-major (M×K for A, K×N for B, M×N for C), but cuBLAS expects column-major.
    // For row-major A (M×K), cuBLAS sees it as K×M column-major: lda = numRowsA
    // For row-major B (K×N), cuBLAS sees it as N×K column-major: ldb = numColsA
    // For row-major C (M×N), cuBLAS sees it as N×M column-major: ldc = numRowsA
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition for both matrices
                                 numColsB, numRowsA, numColsA,            // m=N, n=M, k=K
                                 &alpha,                                  // Scalar alpha
                                 deviceMatrixB, numColsA,                 // B: N×K, lda=K
                                 deviceMatrixA, numRowsA,                 // A: K×M, lda=M
                                 &beta,                                   // Scalar beta
                                 deviceResultMatrix, numRowsA));          // C: N×M, ldc=M
    }
    // For double: Use cublasDgemm (double precision)
    else if constexpr (std::is_same_v<T, double>) {
        CUBLAS_CHECK(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition for both matrices
                                 numColsB, numRowsA, numColsA,            // m=N, n=M, k=K
                                 &alpha,                                  // Scalar alpha
                                 deviceMatrixB, numColsA,                 // B: N×K, lda=K
                                 deviceMatrixA, numRowsA,                 // A: K×M, lda=M
                                 &beta,                                   // Scalar beta
                                 deviceResultMatrix, numRowsA));          // C: N×M, ldc=M
    }
    // If neither float nor double, throw a compile-time error
    else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double types are supported for matrix multiplication");
    }

    // Copy the result matrix from the device (GPU) back to the host (CPU)
    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, byteSizeC, cudaMemcpyDeviceToHost));

    // Clean up: Destroy cuBLAS handle and free the allocated GPU memory
    CUBLAS_CHECK(cublasDestroy(cublasHandle));  // Destroy cuBLAS context
    CUDA_CHECK(cudaFree(deviceMatrixA));        // Free memory for matrix A
    CUDA_CHECK(cudaFree(deviceMatrixB));        // Free memory for matrix B
    CUDA_CHECK(cudaFree(deviceResultMatrix));   // Free memory for result matrix C
}

// Explicit template instantiations for float and double types
template void multiplyMatricesOnGPU<float>(const float*, const float*, float*, int, int, int);
template void multiplyMatricesOnGPU<double>(const double*, const double*, double*, int, int, int);

namespace cuda_kernel {

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

template <typename T>
__global__ void matrixMulTiledKernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ T sharedA[16][17];
    __shared__ T sharedB[16][17];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    T sum = 0;

    for (int phase = 0; phase < (N + 15) / 16; phase++) {
        if (row < M && (phase * 16 + tx) < N) {
            sharedA[ty][tx] = A[row * N + phase * 16 + tx];
        } else {
            sharedA[ty][tx] = 0;
        }

        if ((phase * 16 + ty) < N && col < K) {
            sharedB[ty][tx] = B[(phase * 16 + ty) * K + col];
        } else {
            sharedB[ty][tx] = 0;
        }

        __syncthreads();

        for (int i = 0; i < 16; i++) {
            sum += sharedA[ty][i] * sharedB[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

}  // namespace cuda_kernel

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

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "\n===== Matrix Multiplication Benchmark (" << size << "x" << size << ") =====" << std::endl;

    CUDA_CHECK(cudaEventRecord(start));
    multiplyMatricesNaive(h_A.data(), h_B.data(), h_C_naive.data(), M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
    std::cout << "Naive:  " << naiveTime << " ms" << std::endl;

    CUDA_CHECK(cudaEventRecord(start));
    multiplyMatricesTiled(h_A.data(), h_B.data(), h_C_tiled.data(), M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&tiledTime, start, stop));
    std::cout << "Tiled:  " << tiledTime << " ms (speedup: " << naiveTime / tiledTime << "x)" << std::endl;

    CUDA_CHECK(cudaEventRecord(start));
    multiplyMatricesOnGPU(h_A.data(), h_B.data(), h_C_cublas.data(), M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&cublasTime, start, stop));
    std::cout << "cuBLAS: " << cublasTime << " ms (speedup: " << naiveTime / cublasTime << "x)" << std::endl;

    float maxDiffNaive = 0;
    float maxDiffTiled = 0;
    for (int i = 0; i < M * K; i++) {
        float diffNaive = std::abs(h_C_naive[i] - h_C_cublas[i]);
        float diffTiled = std::abs(h_C_tiled[i] - h_C_cublas[i]);
        if (diffNaive > maxDiffNaive) maxDiffNaive = diffNaive;
        if (diffTiled > maxDiffTiled) maxDiffTiled = diffTiled;
    }
    std::cout << "Max diff (naive vs cuBLAS): " << maxDiffNaive << std::endl;
    std::cout << "Max diff (tiled vs cuBLAS): " << maxDiffTiled << std::endl;

    const float EPSILON = 1e-4f;
    bool passed = (maxDiffNaive < EPSILON && maxDiffTiled < EPSILON);
    std::cout << "Verification: " << (passed ? "PASS" : "FAIL")
              << " (threshold: " << EPSILON << ")" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
