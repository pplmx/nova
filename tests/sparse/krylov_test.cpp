#include "sparse_matrix.hpp"
#include "sparse_ops.hpp"
#include "krylov.hpp"
#include "roofline.hpp"
#include "matrix.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

namespace nova {
    namespace sparse {
        namespace test {

            template <typename T>
            bool approx_equal(T a, T b, T tol = T{1e-5}) {
                return std::abs(a - b) < tol;
            }

            template <typename T>
            class KrylovSolverTest {
            public:
                static bool test_cg_trivial() {
                    std::vector<T> dense = {T{4}, T{1}, T{1}, T{3}};
                    auto matrix = SparseMatrix<T>::FromDense(dense.data(), 2, 2);
                    if (!matrix) {
                        return false;
                    }

                    std::vector<T> b = {T{1}, T{2}};
                    std::vector<T> x(2, T{0});

                    SolverConfig<T> config;
                    config.relative_tolerance = T{1e-8};
                    config.max_iterations = 100;
                    config.verbose = false;

                    ConjugateGradient<T> solver(config);
                    auto result = solver.solve(*matrix, b.data(), x.data());

                    if (!result.converged) {
                        std::cerr << "CG: Failed to converge\n";
                        return false;
                    }

                    T expected_x0 = T{1} / T{11};
                    T expected_x1 = T{7} / T{11};

                    if (!approx_equal(x[0], expected_x0, T{1e-4}) || !approx_equal(x[1], expected_x1, T{1e-4})) {
                        std::cerr << "CG: Solution incorrect: (" << x[0] << ", " << x[1] << ") expected (" << expected_x0 << ", " << expected_x1 << ")\n";
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
                        if (i > 0) {
                            dense[i * n + i - 1] = T{-1};
                        }
                        if (i < n - 1) {
                            dense[i * n + i + 1] = T{-1};
                        }
                    }

                    auto matrix = SparseMatrix<T>::FromDense(dense.data(), n, n);
                    if (!matrix) {
                        return false;
                    }

                    std::vector<T> b(n, T{1});
                    std::vector<T> x(n, T{0});

                    SolverConfig<T> config;
                    config.relative_tolerance = T{1e-8};
                    config.max_iterations = 500;

                    ConjugateGradient<T> solver(config);
                    auto result = solver.solve(*matrix, b.data(), x.data());

                    if (!result.converged) {
                        std::cerr << "CG Laplacian: Failed to converge after " << result.iterations << " iterations\n";
                        return false;
                    }

                    std::vector<T> Ax(n, T{0});
                    // The (pointer, pointer) overload is the DEVICE one: passing host
                    // vectors to it makes cusparse dereference host addresses and the
                    // kernel faults. The host-vector overload stages the data and sizes
                    // Ax itself.
                    spmv(*matrix, x, Ax);
                    T residual = T{0};
                    for (int i = 0; i < n; ++i) {
                        residual = std::max(residual, std::abs(Ax[i] - b[i]));
                    }

                    if (residual > T{1e-4}) {
                        std::cerr << "CG Laplacian: Residual too large: " << residual << "\n";
                        return false;
                    }

                    std::cout << "CG Laplacian: PASSED (converged in " << result.iterations << " iterations)\n";
                    return true;
                }

                static bool test_gmres_trivial() {
                    std::vector<T> dense = {T{4}, T{1}, T{1}, T{3}};
                    auto matrix = SparseMatrix<T>::FromDense(dense.data(), 2, 2);
                    if (!matrix) {
                        return false;
                    }

                    std::vector<T> b = {T{1}, T{2}};
                    std::vector<T> x(2, T{0});

                    SolverConfig<T> config;
                    config.relative_tolerance = T{1e-8};
                    config.max_iterations = 100;
                    config.verbose = false;

                    GMRES<T> solver(config, 5);
                    auto result = solver.solve(*matrix, b.data(), x.data());

                    if (!result.converged) {
                        std::cerr << "GMRES: Failed to converge after " << result.iterations << " iterations\n";
                        return false;
                    }

                    T expected_x0 = T{1} / T{11};
                    T expected_x1 = T{7} / T{11};

                    if (!approx_equal(x[0], expected_x0, T{1e-4}) || !approx_equal(x[1], expected_x1, T{1e-4})) {
                        std::cerr << "GMRES: Solution incorrect: (" << x[0] << ", " << x[1] << ") expected (" << expected_x0 << ", " << expected_x1 << ")\n";
                        return false;
                    }

                    std::cout << "GMRES trivial system: PASSED (converged in " << result.iterations << " iterations)\n";
                    return true;
                }

                static bool test_bicgstab_trivial() {
                    std::vector<T> dense = {T{4}, T{1}, T{1}, T{3}};
                    auto matrix = SparseMatrix<T>::FromDense(dense.data(), 2, 2);
                    if (!matrix) {
                        return false;
                    }

                    std::vector<T> b = {T{1}, T{2}};
                    std::vector<T> x(2, T{0});

                    SolverConfig<T> config;
                    config.relative_tolerance = T{1e-8};
                    config.max_iterations = 100;
                    config.verbose = false;

                    BiCGSTAB<T> solver(config);
                    auto result = solver.solve(*matrix, b.data(), x.data());

                    if (!result.converged) {
                        std::cerr << "BiCGSTAB: Failed to converge after " << result.iterations << " iterations\n";
                        return false;
                    }

                    T expected_x0 = T{1} / T{11};
                    T expected_x1 = T{7} / T{11};

                    if (!approx_equal(x[0], expected_x0, T{1e-4}) || !approx_equal(x[1], expected_x1, T{1e-4})) {
                        std::cerr << "BiCGSTAB: Solution incorrect: (" << x[0] << ", " << x[1] << ") expected (" << expected_x0 << ", " << expected_x1 << ")\n";
                        return false;
                    }

                    std::cout << "BiCGSTAB trivial system: PASSED (converged in " << result.iterations << " iterations)\n";
                    return true;
                }

                static bool run_all() {
                    bool passed = true;
                    passed = passed && test_cg_trivial();
                    passed = passed && test_cg_laplacian();
                    passed = passed && test_gmres_trivial();
                    passed = passed && test_bicgstab_trivial();
                    return passed;
                }
            };

        }  // namespace test
    }  // namespace sparse
}  // namespace nova

int main() {
    bool passed = nova::sparse::test::KrylovSolverTest<double>::run_all();
    return passed ? 0 : 1;
}
