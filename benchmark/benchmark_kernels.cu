#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cuda/benchmark/benchmark.h"
#include "cuda/benchmark/nvtx.h"
#include "cuda/algo/reduce.h"
#include "cuda/algo/scan.h"
#include "cuda/algo/fft/fft.h"
#include "cuda/neural/matmul.h"

namespace {

__global__ void dummy_kernel(float* data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void dummy_compute_kernel(float* data, size_t n, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = sin(val) * cos(val);
        }
        data[idx] = val;
    }
}

float* allocate_device(size_t size) {
    float* ptr = nullptr;
    cudaMalloc(&ptr, size);
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

}  // namespace

static void BM_MemoryH2D(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_data(n, 1.0f);
    float* d_data = allocate_device(bytes);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("H2D_transfer");
        cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(bytes * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_MemoryD2H(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_data(n, 1.0f);
    std::vector<float> h_result(n);
    float* d_data = allocate_device(bytes);
    host_to_device(d_data, h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("D2H_transfer");
        cudaMemcpy(h_result.data(), d_data, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(bytes * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_MemoryD2D(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t bytes = n * sizeof(float);

    float* d_src = allocate_device(bytes);
    float* d_dst = allocate_device(bytes);

    std::vector<float> h_data(n, 1.0f);
    host_to_device(d_src, h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("D2D_transfer");
        device_to_device(d_dst, d_src, n);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(bytes * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_src);
    free_device(d_dst);
}

static void BM_KernelDummy(benchmark::State& state) {
    const size_t n = state.range(0);
    const int threads = 256;

    float* d_data = allocate_device(n * sizeof(float));

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("kernel_launch");
        dummy_kernel<<<(n + threads - 1) / threads, threads>>>(d_data, n, 1.0f);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_KernelCompute(benchmark::State& state) {
    const size_t n = state.range(0);
    const int threads = 256;
    const int iterations = state.range(1);

    float* d_data = allocate_device(n * sizeof(float));

    std::vector<float> h_data(n, 1.0f);
    host_to_device(d_data, h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("kernel_compute");
        dummy_compute_kernel<<<(n + threads - 1) / threads, threads>>>(d_data, n, iterations);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

static void BM_AlgoReduce(benchmark::State& state) {
    const size_t n = state.range(0);

    float* d_data = allocate_device(n * sizeof(float));
    float* d_result = allocate_device(sizeof(float));

    std::vector<float> h_data(n, 1.0f);
    host_to_device(d_data, h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("reduce");
        cuda::algo::reduce(d_data, d_result, n);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
    free_device(d_result);
}

static void BM_AlgoScan(benchmark::State& state) {
    const size_t n = state.range(0);

    float* d_data = allocate_device(n * sizeof(float));
    float* d_result = allocate_device(n * sizeof(float));

    std::vector<float> h_data(n, 1.0f);
    host_to_device(d_data, h_data.data(), n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("scan");
        cuda::algo::inclusive_scan(d_data, d_result, n);
        cudaDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(n * sizeof(float) * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
    free_device(d_result);
}

BENCHMARK(BM_MemoryH2D)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 26}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MemoryD2H)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 26}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MemoryD2D)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 26}})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_KernelDummy)->RangeMultiplier(2)->Ranges({{1 << 10, 1 << 26}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_KernelCompute)->Ranges({{1 << 16, 0}, {1 << 20, 0}})->ArgPair(1000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_AlgoReduce)->RangeMultiplier(2)->Ranges({{1 << 16, 1 << 24}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_AlgoScan)->RangeMultiplier(2)->Ranges({{1 << 16, 1 << 24}})->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
