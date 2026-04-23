#include <gtest/gtest.h>
#include "cuda/benchmark/benchmark.h"
#include "cuda/memory/buffer.h"

class ThroughputTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(ThroughputTest, MemoryCopyThroughputInGBps) {
    constexpr size_t N = 100 * 1024 * 1024;

    cuda::benchmark::BenchmarkOptions options;
    options.compute_throughput = true;
    options.data_size_bytes = N;
    options.measurement_iterations = 3;

    cuda::benchmark::Benchmark bench("memory_throughput", options);

    std::vector<char> host_data(N, 1);

    auto result = bench.run([&]() {
        cuda::memory::Buffer<char> src(N);
        cuda::memory::Buffer<char> dst(N);
        src.copy_from(host_data.data(), N);
        CUDA_CHECK(cudaMemcpy(dst.data(), src.data(), N, cudaMemcpyDeviceToDevice));
    });

    EXPECT_GT(result.throughput_gbps, 0.0);
}

TEST_F(ThroughputTest, ThroughputCalculationHelper) {
    size_t gib = 1024ULL * 1024 * 1024;
    double gbps = cuda::benchmark::compute_throughput_gbps(gib, 1.0);
    EXPECT_NEAR(gbps, 1000.0, 0.01);

    gbps = cuda::benchmark::compute_throughput_gbps(gib, 1000.0);
    EXPECT_NEAR(gbps, 1.0, 0.01);

    gbps = cuda::benchmark::compute_throughput_gbps(gib, 0.5);
    EXPECT_NEAR(gbps, 2000.0, 0.01);
}

TEST_F(ThroughputTest, ThroughputIsZeroForZeroTime) {
    double gbps = cuda::benchmark::compute_throughput_gbps(1024ULL * 1024 * 1024, 0.0);
    EXPECT_EQ(gbps, 0.0);
}

TEST_F(ThroughputTest, BenchmarkWithThroughputReportsGBps) {
    cuda::benchmark::BenchmarkOptions options;
    options.compute_throughput = true;
    options.data_size_bytes = 1024 * 1024;
    options.measurement_iterations = 1;

    cuda::benchmark::Benchmark bench("throughput_test", options);

    auto result = bench.run([]() {
        void* ptr;
        cudaMalloc(&ptr, 1024 * 1024);
        cudaFree(ptr);
    });

    EXPECT_TRUE(result.throughput_gbps >= 0.0);
}