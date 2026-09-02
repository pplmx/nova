#include <gtest/gtest.h>

#include "cuda/neural/tensor_parallel_matmul.h"
#include "cuda/neural/tensor_parallel_profile.h"

using namespace cuda::neural;

// These are CPU-only contracts (buffer-size formulas, profiler structs,
// strategy enum values) — deliberately NO cudaDeviceReset() in SetUp: a reset
// here destroys the CUDA context the shared NcclContext singleton's
// communicators live on (undetectable to the context, per the v2.19
// shared-reset convention), so an earlier suite's initialized communicators
// go stale and a later multi-GPU suite hangs on them (RIL TASK-065 / ISS-004).
class TensorParallelTest : public ::testing::Test {};

TEST_F(TensorParallelTest, RequiredBufferSizeFormula) {
    constexpr size_t size_2gpus = TensorParallelMatmul::required_buffer_size(1024, 1024, 1024, 2);
    constexpr size_t size_4gpus = TensorParallelMatmul::required_buffer_size(1024, 1024, 1024, 4);
    EXPECT_GT(size_2gpus, 0u);
    EXPECT_GT(size_4gpus, 0u);
    EXPECT_EQ(size_2gpus, size_4gpus * 2);
}

TEST_F(TensorParallelTest, RequiredBufferSizeSingleGPU) {
    constexpr size_t size_1gpu = TensorParallelMatmul::required_buffer_size(1024, 1024, 1024, 1);
    EXPECT_EQ(size_1gpu, 0u);
}

TEST_F(TensorParallelTest, TensorParallelProfileCalculation) {
    auto profile = TensorParallelProfiler::profile(32, 128, 768, 2);

    EXPECT_GT(profile.weight_shard_bytes, 0u);
    EXPECT_GT(profile.activation_bytes, 0u);
    EXPECT_GT(profile.gradient_bytes, 0u);
    EXPECT_GT(profile.total_bytes, 0u);
    EXPECT_GE(profile.max_tp_degree, 1);
}

TEST_F(TensorParallelTest, TensorParallelProfileGradientLargerThanActivation) {
    auto profile = TensorParallelProfiler::profile(32, 128, 768, 2);
    EXPECT_GE(profile.gradient_bytes, profile.activation_bytes);
}

TEST_F(TensorParallelTest, EstimateMemoryPerGPU) {
    size_t mem_1gpu = TensorParallelProfiler::estimate_memory_per_gpu(768, 1);
    size_t mem_2gpu = TensorParallelProfiler::estimate_memory_per_gpu(768, 2);
    size_t mem_4gpu = TensorParallelProfiler::estimate_memory_per_gpu(768, 4);

    EXPECT_GT(mem_1gpu, 0u);
    EXPECT_GT(mem_2gpu, 0u);
    EXPECT_GT(mem_4gpu, 0u);

    EXPECT_EQ(mem_2gpu, mem_1gpu / 2);
    EXPECT_EQ(mem_4gpu, mem_1gpu / 4);
}

TEST_F(TensorParallelTest, StrategyEnumValues) {
    EXPECT_EQ(static_cast<int>(TensorParallelStrategy::ColumnParallel), 0);
    EXPECT_EQ(static_cast<int>(TensorParallelStrategy::RowParallel), 1);
}
