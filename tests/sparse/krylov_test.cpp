#include "sparse_matrix.hpp"
#include "sparse_ops.hpp"
#include "krylov.hpp"
#include "roofline.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

namespace nova {
namespace sparse {
namespace test {

template<typename T>
bool approx_equal(T a, T b, T tol = T{1e-5}) {
    return std::abs(a - b) < tol;
}

template<typename T>
class KrylovSolverTest {
public:
    static bool test_cg_trivial() {
        std::vector<T> dense = {
            T{4}, T{1},
            T{1}, T{3}
        };
        auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), 2, 2);
        if (!csr) return false;

        std::vector<T> b = {T{1}, T{2}};
        std::vector<T> x(2, T{0});

        SolverConfig<T> config;
        config.relative_tolerance = T{1e-8};
        config.max_iterations = 100;
        config.verbose = false;

        ConjugateGradient<T> solver(config);
        auto result = solver.solve(*csr, b.data(), x.data());

        if (!result.converged) {
            std::cerr << "CG: Failed to converge\n";
            return false;
        }

        T expected_x0 = T{1} / T{11};
        T expected_x1 = T{7} / T{11};

        if (!approx_equal(x[0], expected_x0, T{1e-4}) ||
            !approx_equal(x[1], expected_x1, T{1e-4})) {
            std::cerr << "CG: Solution incorrect: (" << x[0] << ", " << x[1]
                      << ") expected (" << expected_x0 << ", " << expected_x1 << ")\n";
            return false;
        }

        std::cout << "CG trivial system: PASSED\n";
        return true;
    }

    static bool test_cg_laplacian() {
        const int n = 10;
        std::vector<T> dense(n * n, T{0});

        for (int i = 0; i < n; ++i) {
            dense[i * n + i] = T{4};
            if (i > 0) dense[i * n + i - 1] = T{-1};
            if (i < n - 1) dense[i * n + i + 1] = T{-1};
        }

        auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), n, n);
        if (!csr) return false;

        std::vector<T> b(n, T{1});
        std::vector<T> x(n, T{0});

        SolverConfig<T> config;
        config.relative_tolerance = T{1e-8};
        config.max_iterations = 500;

        ConjugateGradient<T> solver(config);
        auto result = solver.solve(*csr, b.data(), x.data());

        if (!result.converged) {
            std::cerr << "CG Laplacian: Failed to converge after " << result.iterations << " iterations\n";
            return false;
        }

        std::vector<T> Ax(n, T{0});
        sparse_mv(*csr, x.data(), Ax.begin());
        T residual = T{0};
        for (int i = 0; i < n; ++i) {
            residual = std::max(residual, std::abs(Ax[i] - b[i]));
        }

        if (residual > T{1e-4}) {
            std::cerr << "CG Laplacian: Residual too large: " << residual << "\n";
            return false;
        }

        std::cout << "CG Laplacian (" << n << "x" << n << "): PASSED ("
                  << result.iterations << " iterations, rel_res = " << result.relative_residual << ")\n";
        return true;
    }

    static bool test_gmres_non_symmetric() {
        std::vector<T> dense = {
            T{10}, T{1}, T{2},
            T{3},  T{9}, T{1},
            T{1},  T{2}, T{7}
        };
        auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), 3, 3);
        if (!csr) return false;

        std::vector<T> b = {T{1}, T{2}, T{3}};
        std::vector<T> x(3, T{0});

        SolverConfig<T> config;
        config.relative_tolerance = T{1e-8};
        config.max_iterations = 100;

        GMRES<T> solver(config, 10);
        auto result = solver.solve(*csr, b.data(), x.data());

        if (!result.converged) {
            std::cerr << "GMRES: Failed to converge after " << result.iterations << " iterations\n";
            return false;
        }

        std::vector<T> Ax(3, T{0});
        sparse_mv(*csr, x.data(), Ax.begin());
        T max_err = T{0};
        for (int i = 0; i < 3; ++i) {
            max_err = std::max(max_err, std::abs(Ax[i] - b[i]));
        }

        if (max_err > T{1e-3}) {
            std::cerr << "GMRES: Solution error too large: " << max_err << "\n";
            return false;
        }

        std::cout << "GMRES non-symmetric (3x3): PASSED ("
                  << result.iterations << " iterations)\n";
        return true;
    }

    static bool test_bicgstab_non_symmetric() {
        std::vector<T> dense = {
            T{4}, T{1}, T{0},
            T{1}, T{4}, T{1},
            T{0}, T{1}, T{4}
        };
        auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), 3, 3);
        if (!csr) return false;

        std::vector<T> b = {T{1}, T{2}, T{1}};
        std::vector<T> x(3, T{0});

        SolverConfig<T> config;
        config.relative_tolerance = T{1e-8};
        config.max_iterations = 100;

        BiCGSTAB<T> solver(config);
        auto result = solver.solve(*csr, b.data(), x.data());

        if (!result.converged) {
            std::cerr << "BiCGSTAB: Failed to converge after " << result.iterations << " iterations\n";
            return false;
        }

        std::vector<T> Ax(3, T{0});
        sparse_mv(*csr, x.data(), Ax.begin());
        T max_err = T{0};
        for (int i = 0; i < 3; ++i) {
            max_err = std::max(max_err, std::abs(Ax[i] - b[i]));
        }

        if (max_err > T{1e-4}) {
            std::cerr << "BiCGSTAB: Solution error too large: " << max_err << "\n";
            return false;
        }

        std::cout << "BiCGSTAB non-symmetric (3x3): PASSED ("
                  << result.iterations << " iterations)\n";
        return true;
    }

    static bool test_roofline_device_peaks() {
        auto peaks = get_device_peaks();

        if (peaks.fp32_peak_gflops <= 0) {
            std::cerr << "Roofline: Invalid FP32 peak: " << peaks.fp32_peak_gflops << "\n";
            return false;
        }

        if (peaks.memory_bandwidth_gbps <= 0) {
            std::cerr << "Roofline: Invalid memory bandwidth: " << peaks.memory_bandwidth_gbps << "\n";
            return false;
        }

        std::cout << "Roofline device peaks:\n";
        std::cout << "  FP64: " << peaks.fp64_peak_gflops << " GFLOPS\n";
        std::cout << "  FP32: " << peaks.fp32_peak_gflops << " GFLOPS\n";
        std::cout << "  FP16: " << peaks.fp16_peak_gflops << " GFLOPS\n";
        std::cout << "  Bandwidth: " << peaks.memory_bandwidth_gbps << " GB/s\n";
        std::cout << "  CC: " << peaks.compute_capability_major << "."
                  << peaks.compute_capability_minor << "\n";
        std::cout << "  SMs: " << peaks.multiprocessor_count << "\n";

        std::cout << "Roofline device peaks: PASSED\n";
        return true;
    }

    static bool test_arithmetic_intensity() {
        int nnz = 1000;
        int n = 500;
        double ai = spmv_arithmetic_intensity<double>(nnz, n);

        long long flops = 2LL * nnz;
        size_t bytes = static_cast<size_t>(nnz) * sizeof(double) * 2 +
                       static_cast<size_t>(n) * sizeof(double) * 2;
        double expected_ai = static_cast<double>(flops) / static_cast<double>(bytes);

        if (std::abs(ai - expected_ai) > 1e-10) {
            std::cerr << "Arithmetic intensity: Mismatch: " << ai << " vs " << expected_ai << "\n";
            return false;
        }

        std::cout << "Arithmetic intensity (nnz=" << nnz << ", n=" << n << "): "
                  << ai << " FLOPs/byte - PASSED\n";
        return true;
    }

    static bool test_roofline_classification() {
        RooflineAnalyzer analyzer;
        auto peaks = analyzer.device_peaks();

        double spmv_ai = spmv_arithmetic_intensity<double>(1000, 500);

        auto metrics = analyzer.analyze_kernel("SpMV", 2000, 0.1, 8000, Precision::FP32);

        if (metrics.bound == PerformanceBound::UNKNOWN) {
            std::cerr << "Roofline: Classification failed\n";
            return false;
        }

        std::cout << "Roofline classification:\n";
        std::cout << "  AI: " << metrics.arithmetic_intensity << " FLOPs/byte\n";
        std::cout << "  Peak: " << metrics.peak_gflops << " GFLOPS\n";
        std::cout << "  Bound: ";
        switch (metrics.bound) {
            case PerformanceBound::COMPUTE_BOUND: std::cout << "COMPUTE_BOUND\n"; break;
            case PerformanceBound::MEMORY_BOUND: std::cout << "MEMORY_BOUND\n"; break;
            case PerformanceBound::BALANCED: std::cout << "BALANCED\n"; break;
            default: std::cout << "UNKNOWN\n";
        }

        std::cout << "Roofline classification: PASSED\n";
        return true;
    }

    static int run_all() {
        int passed = 0;
        int failed = 0;

        std::cout << "\n=== Krylov Solver Tests (T=" << (sizeof(T) == 4 ? "float" : "double") << ") ===\n";

        if (test_cg_trivial()) ++passed; else ++failed;
        if (test_cg_laplacian()) ++passed; else ++failed;
        if (test_gmres_non_symmetric()) ++passed; else ++failed;
        if (test_bicgstab_non_symmetric()) ++passed; else ++failed;

        std::cout << "\n=== Roofline Tests ===\n";

        if (test_roofline_device_peaks()) ++passed; else ++failed;
        if (test_arithmetic_intensity()) ++passed; else ++failed;
        if (test_roofline_classification()) ++passed; else ++failed;

        std::cout << "\n=== Results ===\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";

        return failed;
    }
};

}
}
}

int main() {
    return nova::sparse::test::KrylovSolverTest<double>::run_all();
}
