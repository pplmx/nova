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

#include <cuda/sparse/krylov.hpp>
#include <cuda/sparse/preconditioner.hpp>
#include <cuda/sparse/reordering.hpp>
#include <cuda/sparse/sparse_concepts.hpp>

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

    // The sparse templates used to accept any T and then fail deep inside the
    // implementation: SparseMatrix<std::string> produced six unrelated errors about
    // narrowing and missing operators. The concepts make the requirement part of
    // the interface. These pin which types are accepted, since a concept that is
    // too narrow would silently reject a legitimate instantiation.
    TEST(SparseHeaderCompileTest, SparseScalarAdmitsTheSupportedScalars) {
        static_assert(SparseScalar<float>);
        static_assert(SparseScalar<double>);
        static_assert(SparseScalar<long double>);

        static_assert(!SparseScalar<int>);
        static_assert(!SparseScalar<unsigned>);
        static_assert(!SparseScalar<bool>);
        static_assert(!SparseScalar<char>);
    }

    TEST(SparseHeaderCompileTest, ConstrainedTemplatesStillInstantiate) {
        // One instantiation per constrained template family, so a constraint that
        // rejects the types the tree actually uses breaks here rather than in a
        // downstream module.
        JacobiPreconditioner<double> jacobi;
        JacobiPreconditioner<float> jacobi_f;
        EXPECT_DOUBLE_EQ(jacobi.omega(), 1.0);
        EXPECT_FLOAT_EQ(jacobi_f.omega(), 1.0f);

        ILUPreconditioner<double> ilu;
        (void)ilu;

        RCMReorderer<double> rcm;
        (void)rcm;

        ConjugateGradient<double> cg;
        GMRES<double> gmres;
        BiCGSTAB<double> bicgstab;
        (void)cg;
        (void)gmres;
        (void)bicgstab;
    }

    TEST(SparseHeaderCompileTest, CppLevelDetectsHostCpp23) {
        // The project builds host code as C++23 and device code as C++20; this
        // macro is the single place that distinction is recorded.
        static_assert(NOVA_CPP23_HOST == 1, "the host build is expected to provide C++23 library features");
    }

    TEST(SparseHeaderCompileTest, RooflineArithmeticIntensity) {
        RooflineAnalyzer analyzer;

        const int nnz = 100;
        const int n = 50;
        const double intensity = analyzer.compute_arithmetic_intensity(
            2LL * nnz, static_cast<size_t>(nnz) * sizeof(float) + static_cast<size_t>(nnz) * sizeof(int) + static_cast<size_t>(n) * sizeof(float) * 2);
        EXPECT_GT(intensity, 0.0);

        // Zero bytes must not divide by zero.
        EXPECT_DOUBLE_EQ(analyzer.compute_arithmetic_intensity(10, 0), 0.0);
    }

    TEST(SparseHeaderCompileTest, SpmvArithmeticIntensityMatchesItsHelper) {
        RooflineAnalyzer analyzer;

        // A 2x2 CSR with two entries: the member function is what used to be
        // uncallable (SparseMatrixCSR was undeclared at that point in the header).
        SparseMatrixCSR<float> csr(std::vector<float> {1.0f, 2.0f}, std::vector<int> {0, 1, 2}, std::vector<int> {0, 1}, 2, 2);

        EXPECT_DOUBLE_EQ(analyzer.compute_arithmetic_intensity_spmv<float>(csr), spmv_arithmetic_intensity<float>(csr.nnz(), csr.num_rows()));
        EXPECT_GT(analyzer.compute_arithmetic_intensity_spmv<float>(csr), 0.0);
    }

    TEST(SparseHeaderCompileTest, ClassificationSplitsOnTheBandwidthCeiling) {
        RooflineAnalyzer analyzer;
        const DevicePeaks peaks = get_device_peaks();

        ASSERT_GT(peaks.memory_bandwidth_gbps, 0.0);
        ASSERT_GT(peaks.fp32_peak_gflops, 0.0);

        // Far below the ridge point: memory bound. Far above: compute bound.
        const double ridge = peaks.fp32_peak_gflops / peaks.memory_bandwidth_gbps;
        EXPECT_EQ(analyzer.classify(ridge * 0.01, peaks.fp32_peak_gflops), PerformanceBound::MEMORY_BOUND);
        EXPECT_EQ(analyzer.classify(ridge * 100.0, peaks.fp32_peak_gflops), PerformanceBound::COMPUTE_BOUND);
    }

    // Regression: roofline.hpp computed the FP64 peak at half the real rate for every
    // architecture from sm_75 through sm_90, and had no test to notice.
    //
    //     case 90/86/89/80/75:  fp64 = SMs * ghz * 16 * 2
    //
    // An A100 (sm_80, 108 SMs, 1.41 GHz) has 32 FP64 cores per SM at 2 flops per FMA:
    //
    //     108 * 32 * 2 * 1.41 = 9.75 TFLOPS   (NVIDIA publishes 9.7)
    //
    // so the table reported 4.87 TFLOPS. The FP32 line in the same case was already right
    // (108 * 128 * 1.41 = 19.5 TFLOPS, matching the published figure), which is how a
    // half-rate FP64 slipped through: one line of a table is easy to check by eye and the
    // line under it is not.
    TEST(RooflinePeaksTest, Fp64PeakIsHalfTheFp32Peak) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "no CUDA device";
        }

        cudaDeviceProp prop {};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

        const auto peaks = nova::sparse::get_device_peaks(0);
        ASSERT_GT(peaks.fp32_peak_gflops, 0.0) << "peak table did not match this architecture";

        // On every architecture this table covers, FP64 vector throughput is exactly half of
        // FP32: the FP64 cores run at the same clock and issue one FMA each, and there are
        // half as many of them. A ratio of 1/4 means the FP64 row used half the core count.
        const double ratio = peaks.fp64_peak_gflops / peaks.fp32_peak_gflops;
        EXPECT_NEAR(ratio, 0.5, 0.05) << "fp64/fp32 = " << ratio << " (" << peaks.fp64_peak_gflops << " vs " << peaks.fp32_peak_gflops << " GFLOPS) on " << prop.name;

        // And the absolute FP64 figure matches the device it queried.
        const double clock_ghz = static_cast<double>(prop.clockRate) / 1e6;
        const double expected = static_cast<double>(prop.multiProcessorCount) * 32.0 * 2.0 * clock_ghz;
        EXPECT_NEAR(peaks.fp64_peak_gflops, expected, 0.2 * expected);
    }

}  // namespace nova::sparse::test
