/**
 * @file sparse_header_compile_test.cpp
 * @brief Compile contracts for the sparse headers that had none
 *
 * Regression: two sparse headers only compiled by accident.
 *
 * - roofline.hpp declared compute_arithmetic_intensity_spmv over
 *   SparseMatrixCSR without including the header that defines it, so including
 *   roofline.hpp on its own failed with "'SparseMatrixCSR' does not name a
 *   type".
 * - nvtx_sparse.hpp left the `namespace nova { namespace sparse { namespace
 *   nvtx {` it had opened unclosed in its NOVA_NVTX branch, so the enabled path
 *   did not compile at all ("expected '}' at end of input"). The default build
 *   never defines that macro, which is why nothing noticed.
 *
 * roofline.hpp is included first and alone below, so a regression breaks this
 * translation unit. The nvtx markers are exercised through the macros, which is
 * the only interface both build modes share; nvtx_sparse.hpp is pulled in via
 * the project's canonical guard (NOVA_NVTX) so the enabled path is compiled by
 * a build that opts in.
 */

#include <cuda/sparse/roofline.hpp>

#include <cuda/sparse/nvtx_sparse.hpp>

#include <gtest/gtest.h>

// Same contract for the GNN and quantization headers. Both used to include
// <cuda/memory/buffer.hpp>, a path that does not exist (the real header is
// buffer.h) and whose Buffer type neither of them uses; each also relied on
// math functions without including <cmath>. Neither directory is part of the
// compiled build, so nothing caught it.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cuda/gnn/attention.hpp>
#include <cuda/gnn/message_passing.hpp>
#include <cuda/gnn/sampling.hpp>
#include <cuda/quantize/quantize_ops.hpp>
#include <cuda/quantize/quantize_tensor.hpp>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cuda/sparse/sparse_matrix.hpp>
#pragma GCC diagnostic pop

namespace nova::sparse::test {

namespace {

// The nvtx macros expand to no-ops unless NVTX is enabled; calling them pins
// both expansions and, in an enabled build, the closed namespace and the
// ScopedRange type.
void emit_nvtx_markers() {
    NOVA_NVTX_MARKER("sparse_probe");
    NOVA_NVTX_SCOPED_RANGE("sparse_probe_range");
}

}  // namespace

TEST(SparseHeaderCompileTest, NvtxMacrosExpandInEitherBuild) {
    EXPECT_NO_THROW(emit_nvtx_markers());
}

TEST(SparseHeaderCompileTest, RooflineArithmeticIntensity) {
    RooflineAnalyzer analyzer;

    const int nnz = 100;
    const int n = 50;
    const double intensity = analyzer.compute_arithmetic_intensity(
        2LL * nnz,
        static_cast<size_t>(nnz) * sizeof(float) +
        static_cast<size_t>(nnz) * sizeof(int) +
        static_cast<size_t>(n) * sizeof(float) * 2);
    EXPECT_GT(intensity, 0.0);

    // Zero bytes must not divide by zero.
    EXPECT_DOUBLE_EQ(analyzer.compute_arithmetic_intensity(10, 0), 0.0);
}

TEST(SparseHeaderCompileTest, SpmvArithmeticIntensityMatchesItsHelper) {
    RooflineAnalyzer analyzer;

    // A 2x2 CSR with two entries: the member function is what used to be
    // uncallable (SparseMatrixCSR was undeclared at that point in the header).
    SparseMatrixCSR<float> csr(std::vector<float>{1.0f, 2.0f},
                              std::vector<int>{0, 1, 2},
                              std::vector<int>{0, 1},
                              2, 2);

    EXPECT_DOUBLE_EQ(analyzer.compute_arithmetic_intensity_spmv<float>(csr),
                     spmv_arithmetic_intensity<float>(csr.nnz(), csr.num_rows()));
    EXPECT_GT(analyzer.compute_arithmetic_intensity_spmv<float>(csr), 0.0);
}

TEST(SparseHeaderCompileTest, ClassificationSplitsOnTheBandwidthCeiling) {
    RooflineAnalyzer analyzer;
    const DevicePeaks peaks = get_device_peaks();

    ASSERT_GT(peaks.memory_bandwidth_gbps, 0.0);
    ASSERT_GT(peaks.fp32_peak_gflops, 0.0);

    // Far below the ridge point: memory bound. Far above: compute bound.
    const double ridge = peaks.fp32_peak_gflops / peaks.memory_bandwidth_gbps;
    EXPECT_EQ(analyzer.classify(ridge * 0.01, peaks.fp32_peak_gflops),
              PerformanceBound::MEMORY_BOUND);
    EXPECT_EQ(analyzer.classify(ridge * 100.0, peaks.fp32_peak_gflops),
              PerformanceBound::COMPUTE_BOUND);
}

}  // namespace nova::sparse::test
