/**
 * @file tensor_parallel_layers_test.cpp
 * @brief Fail-fast disposition tests for the TensorParallelLayers skeletons
 *
 * Milestone v2.20 P2 (TASK-009 / issue-v20-tp-layers-stubs): the v1.3
 * ColumnParallelLayer / RowParallelLayer / TensorParallelMLP wrappers were
 * non-functional scaffolding — they own no weight storage and their forward()
 * bodies passed B = nullptr into the cuBLAS-backed TensorParallelMatmul (a
 * null-pointer crash on the happy path) or returned an empty result. They have
 * zero callers in the tree. Per the library's fail-fast convention they now
 * throw an explicit "not implemented" error on every forward() until a
 * weight-managed implementation lands (DEC-006).
 *
 * These tests pin that contract: constructing the layer objects is safe (it
 * only wires an uninitialized NcclContext reference + sub-matmuls), and each
 * forward() must throw rather than dereference a null weight or return
 * nothing. No GPU is required beyond the CUDA runtime being present.
 */

#include <gtest/gtest.h>

#include "cuda/neural/tensor_parallel_layers.h"

using namespace cuda::neural;

namespace {

// An uninitialized NcclContext is enough to construct the layer wrappers; their
// forward() must fail fast before touching any CUDA state.
cuda::nccl::NcclContext& never_initialized_ctx() {
    static cuda::nccl::NcclContext* ctx = new cuda::nccl::NcclContext();
    return *ctx;
}

}  // namespace

class TensorParallelLayersTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        cudaFree(nullptr);  // Ensure a CUDA context for device-independent calls
    }
};

// ColumnParallelLayer::forward previously passed a null weight pointer to
// cuBLAS; it must now fail fast with an explicit error.
TEST_F(TensorParallelLayersTest, ColumnParallelLayerForwardRejects) {
    ColumnParallelLayer layer(never_initialized_ctx(), /*hidden_dim=*/64, /*tp_degree=*/2);
    EXPECT_THROW(layer.forward(nullptr, nullptr, /*batch=*/1, /*seq=*/8),
                 std::exception)
        << "ColumnParallelLayer::forward must fail fast (unimplemented: no "
           "weight storage), not dereference a null weight";
}

// RowParallelLayer::forward previously passed a null weight pointer to cuBLAS.
TEST_F(TensorParallelLayersTest, RowParallelLayerForwardRejects) {
    RowParallelLayer layer(never_initialized_ctx(), /*hidden_dim=*/64, /*tp_degree=*/2);
    EXPECT_THROW(layer.forward(nullptr, nullptr, /*batch=*/1, /*seq=*/8),
                 std::exception)
        << "RowParallelLayer::forward must fail fast (unimplemented: no weight "
           "storage), not dereference a null weight";
}

// TensorParallelMLP::forward previously had an empty body (silent no-op).
TEST_F(TensorParallelLayersTest, TensorParallelMLPForwardRejects) {
    TensorParallelMLP mlp(never_initialized_ctx(), /*hidden_dim=*/64,
                          /*intermediate_size=*/128, /*tp_degree=*/2);
    EXPECT_THROW(mlp.forward(nullptr, nullptr, /*batch=*/1, /*seq=*/8),
                 std::exception)
        << "TensorParallelMLP::forward must fail fast (unimplemented) instead "
           "of silently returning with no output";
}
