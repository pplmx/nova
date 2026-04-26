#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "cuda/benchmark/benchmark.h"
#include "cuda/benchmark/nvtx.h"
#include "cuda/algo/reduce.h"
#include "cuda/fft/fft.h"
#include "cuda/neural/matmul.h"
#include "cuda/memory/buffer.h"
#include "parallel/scan.h"
#include "parallel/sort.h"

namespace {

constexpr int BLOCK_SIZE = 256;

__global__ void init_kernel(float* data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void scale_kernel(float* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

float* allocate_device(size_t size) {
    float* ptr = nullptr;
    cudaMalloc(&ptr, size * sizeof(float));
    return ptr;
}

void free_device(float* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void host_to_device(float* dst, const float* src, size_t n) {
    cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice);
}

void device_to_host(float* dst, const float* src, size_t n) {
    cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost);
}

void device_to_device(float* dst, const float* src, size_t n) {
    cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

void init_device_array(float* d_data, size_t n, float value) {
    init_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, n, value);
    cudaDeviceSynchronize();
}

}  // namespace

static void BM_MemoryH2D(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_data(n, 1.0f);
    float* d_data = allocate_device(n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("H2D_transfer");
        cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(bytes * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_MemoryD2H(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_result(n);
    float* d_data = allocate_device(n);
    init_device_array(d_data, n, 1.0f);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("D2H_transfer");
        cudaMemcpy(h_result.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(bytes * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_MemoryD2D(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t bytes = n * sizeof(float);

    float* d_src = allocate_device(n);
    float* d_dst = allocate_device(n);
    init_device_array(d_src, n, 1.0f);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("D2D_transfer");
        device_to_device(d_dst, d_src, n);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(bytes * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_src);
    free_device(d_dst);
}

static void BM_AlgoReduceSum(benchmark::State& state) {
    const size_t n = state.range(0);

    float* d_data = allocate_device(n);
    init_device_array(d_data, n, 1.0f);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("reduce_sum");
        float result = cuda::algo::reduce_sum(d_data, n);
        (void)result;
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_AlgoReduceMax(benchmark::State& state) {
    const size_t n = state.range(0);

    float* d_data = allocate_device(n);
    init_device_array(d_data, n, 1.0f);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("reduce_max");
        float result = cuda::algo::reduce_max(d_data, n);
        (void)result;
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_AlgoReduceOptimized(benchmark::State& state) {
    const size_t n = state.range(0);

    float* d_data = allocate_device(n);
    init_device_array(d_data, n, 1.0f);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("reduce_sum_optimized");
        float result = cuda::algo::reduce_sum_optimized(d_data, n);
        (void)result;
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_NeuralMatmul(benchmark::State& state) {
    const int m = state.range(0);
    const int k = state.range(1);
    const int n = state.range(2);

    float* d_A = allocate_device(m * k);
    float* d_B = allocate_device(k * n);
    float* d_C = allocate_device(m * n);

    std::vector<float> h_A(m * k, 1.0f);
    std::vector<float> h_B(k * n, 1.0f);
    host_to_device(d_A, h_A.data(), m * k);
    host_to_device(d_B, h_B.data(), k * n);

    cuda::neural::MatmulOptions options;
    options.handle = cuda::neural::get_cublas_handle();

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("matmul");
        cuda::neural::matmul(d_A, d_B, d_C, m, n, k, options);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t((m * k + k * n + m * n) * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(m * n * state.iterations()));

    free_device(d_A);
    free_device(d_B);
    free_device(d_C);
}

static void BM_NeuralMatmulBatch(benchmark::State& state) {
    const int batch = state.range(0);
    const int m = 64;
    const int k = 64;
    const int n = 64;

    float* d_A = allocate_device(batch * m * k);
    float* d_B = allocate_device(batch * k * n);
    float* d_C = allocate_device(batch * m * n);

    std::vector<float> h_A(batch * m * k, 1.0f);
    std::vector<float> h_B(batch * k * n, 1.0f);
    host_to_device(d_A, h_A.data(), batch * m * k);
    host_to_device(d_B, h_B.data(), batch * k * n);

    cuda::neural::MatmulOptions options;
    options.handle = cuda::neural::get_cublas_handle();

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("matmul_batch");
        cuda::neural::matmul_batch(d_A, d_B, d_C, batch, m, n, k, options);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t((batch * m * k + batch * k * n + batch * m * n) * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(batch * m * n * state.iterations()));

    free_device(d_A);
    free_device(d_B);
    free_device(d_C);
}

static void BM_FFTForward(benchmark::State& state) {
    const size_t n = state.range(0);

    float* d_data = allocate_device(n * 2);
    init_device_array(d_data, n * 2, 1.0f);

    cuda::fft::FFTPlan plan(n, cuda::fft::Direction::Forward, cuda::fft::TransformType::RealToComplex);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("fft_forward");
        plan.execute(d_data, d_data);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * 2 * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_FFTInverse(benchmark::State& state) {
    const size_t n = state.range(0);

    float* d_data = allocate_device(n * 2);
    init_device_array(d_data, n * 2, 1.0f);

    cuda::fft::FFTPlan plan(n, cuda::fft::Direction::Inverse, cuda::fft::TransformType::ComplexToReal);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("fft_inverse");
        plan.execute(d_data, d_data);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * 2 * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_AlgoScanInclusive(benchmark::State& state) {
    const size_t n = state.range(0);

    if (n > MAX_SCAN_SIZE) {
        state.SkipWithError("Scan size exceeds maximum supported size");
        return;
    }

    cuda::memory::Buffer<float> input(n);
    cuda::memory::Buffer<float> output(n);

    std::vector<float> h_data(n, 1.0f);
    input.copy_from(h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("scan_inclusive");
        cuda::algo::inclusiveScan(input, output, n);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));
}

static void BM_AlgoScanExclusive(benchmark::State& state) {
    const size_t n = state.range(0);

    if (n > MAX_SCAN_SIZE) {
        state.SkipWithError("Scan size exceeds maximum supported size");
        return;
    }

    cuda::memory::Buffer<float> input(n);
    cuda::memory::Buffer<float> output(n);

    std::vector<float> h_data(n, 1.0f);
    input.copy_from(h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("scan_exclusive");
        cuda::algo::exclusiveScan(input, output, n);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));
}

static void BM_SortOddEven(benchmark::State& state) {
    const size_t n = state.range(0);

    cuda::memory::Buffer<float> input(n);
    cuda::memory::Buffer<float> output(n);

    std::vector<float> h_data(n);
    std::iota(h_data.begin(), h_data.end(), 0.0f);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(h_data.begin(), h_data.end(), g);
    input.copy_from(h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("sort_odd_even");
        cuda::parallel::oddEvenSort(input, output, n);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));
}

static void BM_SortBitonic(benchmark::State& state) {
    const size_t n = state.range(0);

    cuda::memory::Buffer<float> input(n);
    cuda::memory::Buffer<float> output(n);

    std::vector<float> h_data(n);
    std::iota(h_data.begin(), h_data.end(), 0.0f);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(h_data.begin(), h_data.end(), g);
    input.copy_from(h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("sort_bitonic");
        cuda::parallel::bitonicSort(input, output, n);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));
}

BENCHMARK(BM_MemoryH2D)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 24}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MemoryD2H)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 24}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MemoryD2D)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 24}})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_AlgoReduceSum)->RangeMultiplier(2)->Ranges({{1 << 16, 1 << 24}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_AlgoReduceMax)->RangeMultiplier(2)->Ranges({{1 << 16, 1 << 24}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_AlgoReduceOptimized)->RangeMultiplier(2)->Ranges({{1 << 16, 1 << 24}})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_AlgoScanInclusive)->RangeMultiplier(2)->Ranges({{256, 1024}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_AlgoScanExclusive)->RangeMultiplier(2)->Ranges({{256, 1024}})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_NeuralMatmul)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_NeuralMatmulBatch)
    ->Args({1})
    ->Args({8})
    ->Args({32})
    ->Args({128})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_FFTForward)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 20}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FFTInverse)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 20}})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SortOddEven)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 16}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SortBitonic)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 16}})->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
