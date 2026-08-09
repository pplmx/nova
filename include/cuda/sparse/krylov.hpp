/**
 * @file krylov.hpp
 * @brief Krylov subspace iterative solvers
 * @defgroup krylov Krylov Subspace Solvers
 * @ingroup sparse
 *
 * Provides iterative solvers for sparse linear systems Ax = b:
 * - CG (Conjugate Gradient) for symmetric positive definite matrices
 * - GMRES (Generalized Minimal Residual) for non-symmetric matrices
 * - BiCGSTAB (Biconjugate Gradient Stabilized) for non-symmetric matrices
 *
 * Example usage:
 * @code
 * SolverConfig config{.max_iterations = 1000, .tolerance = 1e-6};
 * auto result = cg(A, b, x, config);
 * if (result.converged) {
 *     std::cout << "Converged in " << result.iterations << " iterations\n";
 * }
 * @endcode
 *
 * @note Time complexity: O(nnz * iterations)
 * @note Requires preconditioner for difficult problems
 * @see preconditioner.hpp For preconditioning support
 * @see solver_workspace.hpp For workspace management
 */

#ifndef NOVA_CUDA_SPARSE_KRYLOV_HPP
#define NOVA_CUDA_SPARSE_KRYLOV_HPP

#include "sparse_matrix.hpp"
#include "sparse_ops.hpp"
#include "matrix.hpp"
#include "preconditioner.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <memory>

namespace nova {
namespace sparse {

/**
 * @brief Solver error codes
 * @enum SolverError
 * @ingroup krylov
 */
enum class SolverError {
    /** @brief Solver completed successfully */
    SUCCESS = 0,

    /** @brief Maximum iterations reached without convergence */
    MAX_ITERATIONS,

    /** @brief Numerical breakdown occurred */
    BREAKDOWN,

    /** @brief Invalid or malformed matrix */
    INVALID_MATRIX,

    /** @brief Failed to meet convergence tolerance */
    CONVERGENCE_FAILURE,

    /** @brief Preconditioner failed or is invalid */
    PRECONDITIONER_ERROR
};

/**
 * @brief Result of iterative solver
 * @struct SolverResult
 * @tparam T Element type
 * @ingroup krylov
 */
template<typename T>
struct SolverResult {
    /** @brief Whether solver converged */
    bool converged = false;

    /** @brief Number of iterations performed */
    int iterations = 0;

    /** @brief Final residual norm */
    T residual_norm = T{0};

    /** @brief Relative residual ||r||/||b|| */
    T relative_residual = T{0};
    SolverError error_code = SolverError::SUCCESS;
    std::vector<T> residual_history;
};

template<typename T>
struct SolverConfig {
    T relative_tolerance = T{1e-6};
    T absolute_tolerance = T{1e-10};
    int max_iterations = 1000;
    bool verbose = false;
};

namespace detail {
    template<typename T>
    T dot_product(const T* a, const T* b, int n) {
        T result = T{0};
        for (int i = 0; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    template<typename T>
    void axpby(T a, const T* x, T b, const T* y, T* z, int n) {
        for (int i = 0; i < n; ++i) {
            z[i] = a * x[i] + b * y[i];
        }
    }

    template<typename T>
    void axpy(T a, const T* x, T* y, int n) {
        for (int i = 0; i < n; ++i) {
            y[i] += a * x[i];
        }
    }

    template<typename T>
    void copy(const T* src, T* dst, int n) {
        for (int i = 0; i < n; ++i) {
            dst[i] = src[i];
        }
    }

    template<typename T>
    void scale(T* x, T alpha, int n) {
        for (int i = 0; i < n; ++i) {
            x[i] *= alpha;
        }
    }

    template<typename T>
    T norm2(const T* x, int n) {
        T sum = T{0};
        for (int i = 0; i < n; ++i) {
            sum += x[i] * x[i];
        }
        return std::sqrt(sum);
    }

    template<typename T>
    void fill(T* x, T value, int n) {
        for (int i = 0; i < n; ++i) {
            x[i] = value;
        }
    }
}

template<typename T>
class KrylovSolver {
public:
    explicit KrylovSolver(const SolverConfig<T>& config = {}) : config_(config) {}
    virtual ~KrylovSolver() = default;

    virtual SolverResult<T> solve(const SparseMatrix<T>& A, const T* b, T* x) = 0;

    void set_preconditioner(std::unique_ptr<Preconditioner<T>> prec) {
        preconditioner_ = std::move(prec);
    }

    bool has_preconditioner() const { return preconditioner_ != nullptr; }

protected:
    SolverConfig<T> config_;
    std::unique_ptr<Preconditioner<T>> preconditioner_;
};

template<typename T>
class ConjugateGradient : public KrylovSolver<T> {
public:
    using KrylovSolver<T>::KrylovSolver;

    SolverResult<T> solve(const SparseMatrix<T>& A, const T* b, T* x) override {
        SolverResult<T> result;
        const int n = A.rows();

        if (A.rows() != A.cols()) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        if (this->preconditioner_ && !has_preconditioner_) {
            this->preconditioner_->setup(A);
            has_preconditioner_ = true;
        }

        memory::Buffer<T> d_b(n), d_x(n), d_r(n), d_z(n), d_p(n), d_Ap(n);
        std::vector<T> h_x(n, T{0}), h_r(n), h_z(n), h_p(n), h_Ap(n);

        d_b.copy_from(b, n);
        d_x.fill(T{0});

        memory::Buffer<T> d_temp(n);
        spmv(A, d_x.data(), d_temp.data());
        std::vector<T> h_temp(n);
        d_temp.copy_to(h_temp.data(), n);

        for (int i = 0; i < n; ++i) {
            h_r[i] = b[i] - h_temp[i];
        }

        T b_norm = detail::norm2(b, n);
        if (b_norm < std::numeric_limits<T>::epsilon()) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        if (this->preconditioner_) {
            this->preconditioner_->apply(h_r.data(), h_z.data());
        } else {
            h_z = h_r;
        }

        h_p = h_z;
        T r_dot_old = detail::dot_product(h_r.data(), h_z.data(), n);

        result.residual_history.reserve(this->config_.max_iterations);

        for (int iter = 0; iter < this->config_.max_iterations; ++iter) {
            d_p.copy_from(h_p.data(), n);
            d_Ap.fill(T{0});
            spmv(A, d_p.data(), d_Ap.data());
            d_Ap.copy_to(h_Ap.data(), n);

            T p_Ap = detail::dot_product(h_p.data(), h_Ap.data(), n);
            if (std::abs(p_Ap) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T alpha = r_dot_old / p_Ap;

            for (int i = 0; i < n; ++i) {
                h_x[i] += alpha * h_p[i];
                h_r[i] -= alpha * h_Ap[i];
            }

            if (this->preconditioner_) {
                this->preconditioner_->apply(h_r.data(), h_z.data());
            } else {
                h_z = h_r;
            }

            T r_dot_new = detail::dot_product(h_r.data(), h_z.data(), n);
            T residual = detail::norm2(h_r.data(), n);

            result.residual_history.push_back(residual);
            result.relative_residual = residual / b_norm;

            if (this->config_.verbose) {
                std::printf("CG iter %d: residual = %.6e, relative = %.6e\n",
                           iter, residual, result.relative_residual);
            }

            if (result.relative_residual < this->config_.relative_tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                result.residual_norm = residual;
                result.error_code = SolverError::SUCCESS;
                for (int i = 0; i < n; ++i) {
                    x[i] = h_x[i];
                }
                return result;
            }

            T beta = r_dot_new / r_dot_old;
            for (int i = 0; i < n; ++i) {
                h_p[i] = h_z[i] + beta * h_p[i];
            }

            r_dot_old = r_dot_new;
        }

        result.iterations = this->config_.max_iterations;
        result.residual_norm = detail::norm2(h_r.data(), n);
        result.error_code = SolverError::MAX_ITERATIONS;
        for (int i = 0; i < n; ++i) {
            x[i] = h_x[i];
        }
        return result;
    }

private:
    bool has_preconditioner_ = false;
};

template<typename T>
class GMRES : public KrylovSolver<T> {
public:
    GMRES(const SolverConfig<T>& config = {}, int restart = 50)
        : KrylovSolver<T>(config), restart_(restart) {}

    SolverResult<T> solve(const SparseMatrix<T>& A, const T* b, T* x) override {
        SolverResult<T> result;
        const int n = A.rows();

        if (A.rows() != A.cols()) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        std::vector<T> h_b(n);
        for (int i = 0; i < n; ++i) {
            h_b[i] = b[i];
        }

        T b_norm = detail::norm2(b, n);
        if (b_norm < std::numeric_limits<T>::epsilon()) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        // Running approximate solution, seeded from the caller's initial guess.
        std::vector<T> h_x(n);
        for (int i = 0; i < n; ++i) {
            h_x[i] = x[i];
        }

        const int m = restart_;
        if (m <= 0) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        std::vector<T> r(n), vj(n), w(n);

        // r = b - A*x0
        spmv(A, h_x, w);
        for (int i = 0; i < n; ++i) {
            r[i] = h_b[i] - w[i];
        }

        T beta = detail::norm2(r.data(), n);
        result.relative_residual = beta / b_norm;
        result.residual_history.push_back(result.relative_residual);
        result.residual_norm = beta;

        if (result.relative_residual < this->config_.relative_tolerance) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        int total_iterations = 0;

        for (int outer = 0; outer * m < this->config_.max_iterations; ++outer) {
            // Arnoldi basis: V is (m+1) x n, H is (m+1) x m (column-major rows).
            std::vector<T> V((m + 1) * n, T{0});
            std::vector<T> H((m + 1) * m, T{0});

            for (int i = 0; i < n; ++i) {
                V[i] = r[i] / beta;
            }

            std::vector<T> g(m + 1, T{0});
            std::vector<T> cs(m, T{0}), sn(m, T{0});
            g[0] = beta;

            for (int j = 0; j < m && total_iterations < this->config_.max_iterations; ++j) {
                for (int i = 0; i < n; ++i) {
                    vj[i] = V[j * n + i];
                }
                spmv(A, vj, w);  // w = A v_j

                // Arnoldi / modified Gram-Schmidt with one reorthogonalization
                // pass (robust against floating-point loss of orthogonality).
                for (int pass = 0; pass < 2; ++pass) {
                    for (int i = 0; i <= j; ++i) {
                        T hij = detail::dot_product(w.data(), V.data() + i * n, n);
                        H[i * m + j] += hij;
                        for (int k = 0; k < n; ++k) {
                            w[k] -= hij * V[i * n + k];
                        }
                    }
                }

                T h_j1j = detail::norm2(w.data(), n);
                H[(j + 1) * m + j] = h_j1j;

                if (h_j1j > std::numeric_limits<T>::epsilon()) {
                    for (int i = 0; i < n; ++i) {
                        V[(j + 1) * n + i] = w[i] / h_j1j;
                    }
                }
                // h_j1j == 0 is the "happy breakdown": the Krylov space is
                // invariant, and the current H[0..j][0..j] + g solve exactly.

                // Apply prior Givens rotations to column j.
                for (int i = 0; i < j; ++i) {
                    T tmp = cs[i] * H[i * m + j] + sn[i] * H[(i + 1) * m + j];
                    H[(i + 1) * m + j] = -sn[i] * H[i * m + j] + cs[i] * H[(i + 1) * m + j];
                    H[i * m + j] = tmp;
                }

                // New Givens rotation zeroing H[j+1][j].
                T hj = H[j * m + j];
                T hnext = H[(j + 1) * m + j];
                T denom = std::sqrt(hj * hj + hnext * hnext);
                T c, s;
                if (denom > std::numeric_limits<T>::epsilon()) {
                    c = hj / denom;
                    s = hnext / denom;
                } else {
                    c = T{1};
                    s = T{0};
                }
                {
                    T tmp = c * H[j * m + j] + s * H[(j + 1) * m + j];
                    H[(j + 1) * m + j] = -s * H[j * m + j] + c * H[(j + 1) * m + j];
                    H[j * m + j] = tmp;
                }
                {
                    T gtmp = c * g[j] + s * g[j + 1];
                    g[j + 1] = -s * g[j] + c * g[j + 1];
                    g[j] = gtmp;
                }
                cs[j] = c;
                sn[j] = s;

                ++total_iterations;
                result.relative_residual = std::abs(g[j + 1]) / b_norm;
                result.residual_history.push_back(result.relative_residual);
                result.residual_norm = std::abs(g[j + 1]);

                if (this->config_.verbose) {
                    std::printf("GMRES iter %d: relative residual = %.6e\n",
                               total_iterations, result.relative_residual);
                }

                if (result.relative_residual < this->config_.relative_tolerance) {
                    // Solve upper-triangular H[0..j][0..j] y = g[0..j].
                    std::vector<T> y(j + 1, T{0});
                    for (int k = j; k >= 0; --k) {
                        T sum = g[k];
                        for (int i = k + 1; i <= j; ++i) {
                            sum -= H[k * m + i] * y[i];
                        }
                        y[k] = sum / H[k * m + k];
                    }
                    for (int i = 0; i < n; ++i) {
                        x[i] = h_x[i];
                    }
                    for (int k = 0; k <= j; ++k) {
                        for (int i = 0; i < n; ++i) {
                            x[i] += y[k] * V[k * n + i];
                        }
                    }

                    result.converged = true;
                    result.iterations = total_iterations;
                    result.error_code = SolverError::SUCCESS;
                    return result;
                }
            }

            // Restart: solve the full m x m system, update the running solution.
            std::vector<T> y(m, T{0});
            for (int k = m - 1; k >= 0; --k) {
                T sum = g[k];
                for (int i = k + 1; i < m; ++i) {
                    sum -= H[k * m + i] * y[i];
                }
                if (std::abs(H[k * m + k]) > std::numeric_limits<T>::epsilon()) {
                    y[k] = sum / H[k * m + k];
                } else {
                    y[k] = T{0};
                }
            }
            for (int k = 0; k < m; ++k) {
                for (int i = 0; i < n; ++i) {
                    h_x[i] += y[k] * V[k * n + i];
                }
            }

            // r = b - A*x
            spmv(A, h_x, w);
            for (int i = 0; i < n; ++i) {
                r[i] = h_b[i] - w[i];
            }
            beta = detail::norm2(r.data(), n);
            result.relative_residual = beta / b_norm;
            result.residual_history.push_back(result.relative_residual);
            result.residual_norm = beta;

            if (this->config_.verbose) {
                std::printf("GMRES restart %d: relative residual = %.6e\n",
                           outer, result.relative_residual);
            }

            if (result.relative_residual < this->config_.relative_tolerance) {
                for (int i = 0; i < n; ++i) {
                    x[i] = h_x[i];
                }
                result.converged = true;
                result.iterations = total_iterations;
                result.error_code = SolverError::SUCCESS;
                return result;
            }

            if (total_iterations >= this->config_.max_iterations) {
                break;
            }
        }

        result.iterations = total_iterations;
        result.residual_norm = result.residual_history.empty()
                                   ? T{0}
                                   : result.residual_history.back();
        result.error_code = SolverError::MAX_ITERATIONS;
        for (int i = 0; i < n; ++i) {
            x[i] = h_x[i];
        }
        return result;
    }

private:
    int restart_;
};

template<typename T>
class GMRESGPU : public KrylovSolver<T> {
public:
    GMRESGPU(const SolverConfig<T>& config = {}, int restart = 50)
        : KrylovSolver<T>(config), restart_(restart) {}

    SolverResult<T> solve(const SparseMatrix<T>& A, const T* b, T* x) {
        SolverResult<T> result;
        const int n = A.rows();

        if (A.rows() != A.cols()) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        memory::Buffer<T> d_b(n), d_x(n);
        std::vector<T> h_x(n, T{0}), h_r(n), h_w(n);

        d_b.copy_from(b, n);
        d_x.fill(T{0});

        memory::Buffer<T> d_temp(n);
        spmv(A, d_x.data(), d_temp.data());
        d_temp.copy_to(h_w.data(), n);

        for (int i = 0; i < n; ++i) {
            h_r[i] = b[i] - h_w[i];
        }

        T beta = detail::norm2(h_r.data(), n);
        T b_norm = detail::norm2(b, n);

        if (b_norm < std::numeric_limits<T>::epsilon()) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        int total_iterations = 0;
        std::vector<T> residual_history;

        for (int outer = 0; outer < this->config_.max_iterations / restart_; ++outer) {
            std::vector<T> V((restart_ + 1) * n);
            std::vector<T> H((restart_ + 1) * restart_, T{0});
            std::vector<T> s(restart_ + 1), cs(restart_), sn(restart_);

            for (int i = 0; i < n; ++i) {
                V[i] = h_r[i] / beta;
            }

            s[0] = beta;
            for (int i = 1; i <= restart_; ++i) {
                s[i] = T{0};
            }

            for (int j = 0; j < restart_ && total_iterations < this->config_.max_iterations; ++j) {
                memory::Buffer<T> d_v(n), d_w(n);
                d_v.copy_from(&V[j * n], n);
                d_w.fill(T{0});
                spmv(A, d_v.data(), d_w.data());
                d_w.copy_to(&V[(j + 1) * n], n);

                for (int i = 0; i <= j; ++i) {
                    T dot = T{0};
                    for (int k = 0; k < n; ++k) {
                        dot += V[(j + 1) * n + k] * V[i * n + k];
                    }
                    H[i * restart_ + j] = dot;
                    for (int k = 0; k < n; ++k) {
                        V[(j + 1) * n + k] -= H[i * restart_ + j] * V[i * n + k];
                    }
                }

                T h_ij = T{0};
                for (int i = 0; i < n; ++i) {
                    h_ij += V[(j + 1) * n + i] * V[(j + 1) * n + i];
                }
                h_ij = std::sqrt(h_ij);

                bool breakdown = h_ij < std::numeric_limits<T>::epsilon();
                if (breakdown) {
                    h_ij = T{1};
                    result.error_code = SolverError::BREAKDOWN;
                }

                for (int i = 0; i < n; ++i) {
                    V[(j + 1) * n + i] /= h_ij;
                }

                H[(j + 1) * restart_ + j] = h_ij;

                for (int k = 0; k < j; ++k) {
                    T c = cs[k];
                    T s_k = sn[k];
                    T h_kj = H[k * restart_ + j];
                    T h_k1j = H[(k + 1) * restart_ + j];
                    H[k * restart_ + j] = c * h_kj + s_k * h_k1j;
                    H[(k + 1) * restart_ + j] = -s_k * h_kj + c * h_k1j;
                }

                T c = H[j * restart_ + j] / std::sqrt(H[j * restart_ + j] * H[j * restart_ + j] + h_ij * h_ij);
                T s_k = h_ij / std::sqrt(H[j * restart_ + j] * H[j * restart_ + j] + h_ij * h_ij);
                cs[j] = c;
                sn[j] = s_k;

                H[j * restart_ + j] = c * H[j * restart_ + j] + s_k * h_ij;
                H[(j + 1) * restart_ + j] = T{0};

                s[j + 1] = -s_k * s[j];
                s[j] = c * s[j];

                T residual = std::abs(s[j + 1]);
                residual_history.push_back(residual);

                if (this->config_.verbose) {
                    std::printf("GMRES(%d) iter %d: residual = %.6e\n", restart_, total_iterations, residual);
                }

                ++total_iterations;

                if (residual < this->config_.relative_tolerance * b_norm) {
                    std::vector<T> y(j + 1);
                    for (int k = 0; k <= j; ++k) {
                        y[k] = s[k];
                    }

                    for (int k = j - 1; k >= 0; --k) {
                        for (int i = k + 1; i <= j; ++i) {
                            y[k] -= H[k * restart_ + i] * y[i];
                        }
                        y[k] /= H[k * restart_ + k];
                    }

                    for (int k = 0; k <= j; ++k) {
                        for (int i = 0; i < n; ++i) {
                            h_x[i] += y[k] * V[k * n + i];
                        }
                    }

                    result.converged = true;
                    result.iterations = total_iterations;
                    result.residual_norm = residual;
                    result.relative_residual = residual / b_norm;
                    result.residual_history = std::move(residual_history);
                    result.error_code = SolverError::SUCCESS;
                    for (int i = 0; i < n; ++i) {
                        x[i] = h_x[i];
                    }
                    return result;
                }
            }

            std::vector<T> y(restart_);
            for (int k = 0; k < restart_; ++k) {
                y[k] = s[k];
            }

            for (int k = restart_ - 1; k >= 0; --k) {
                if (std::abs(H[k * restart_ + k]) < std::numeric_limits<T>::epsilon()) {
                    continue;
                }
                for (int i = k + 1; i < restart_; ++i) {
                    y[k] -= H[k * restart_ + i] * y[i];
                }
                y[k] /= H[k * restart_ + k];
            }

            for (int k = 0; k < restart_; ++k) {
                for (int i = 0; i < n; ++i) {
                    h_x[i] += y[k] * V[k * n + i];
                }
            }

            d_x.copy_from(h_x.data(), n);
            spmv(A, d_x.data(), d_temp.data());
            d_temp.copy_to(h_w.data(), n);

            for (int i = 0; i < n; ++i) {
                h_r[i] = b[i] - h_w[i];
            }

            beta = detail::norm2(h_r.data(), n);

            if (total_iterations >= this->config_.max_iterations) {
                break;
            }
        }

        result.iterations = total_iterations;
        result.residual_norm = beta;
        result.residual_history = std::move(residual_history);
        result.error_code = SolverError::MAX_ITERATIONS;
        for (int i = 0; i < n; ++i) {
            x[i] = h_x[i];
        }
        return result;
    }

private:
    int restart_;
};

template<typename T>
class BiCGSTAB : public KrylovSolver<T> {
public:
    using KrylovSolver<T>::KrylovSolver;

    SolverResult<T> solve(const SparseMatrix<T>& A, const T* b, T* x) override {
        SolverResult<T> result;
        const int n = A.rows();

        if (A.rows() != A.cols()) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        memory::Buffer<T> d_b(n), d_x(n), d_s(n), d_p(n), d_temp(n);
        std::vector<T> h_x(n, T{0}), h_r(n), h_r_tilde(n), h_p(n), h_s(n), h_t(n), h_temp(n);

        d_b.copy_from(b, n);
        d_x.fill(T{0});

        spmv(A, d_x.data(), d_temp.data());
        d_temp.copy_to(h_temp.data(), n);

        for (int i = 0; i < n; ++i) {
            h_r[i] = b[i] - h_temp[i];
        }

        h_r_tilde = h_r;

        T b_norm = detail::norm2(b, n);
        if (b_norm < std::numeric_limits<T>::epsilon()) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        h_p = h_r;

        T r_r_tilde = detail::dot_product(h_r.data(), h_r_tilde.data(), n);

        result.residual_history.reserve(this->config_.max_iterations);

        for (int iter = 0; iter < this->config_.max_iterations; ++iter) {
            d_p.copy_from(h_p.data(), n);
            d_temp.fill(T{0});
            spmv(A, d_p.data(), d_temp.data());
            d_temp.copy_to(h_temp.data(), n);

            T p_Ap = detail::dot_product(h_p.data(), h_temp.data(), n);

            if (std::abs(r_r_tilde) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T alpha = r_r_tilde / p_Ap;

            for (int i = 0; i < n; ++i) {
                h_s[i] = h_r[i] - alpha * h_temp[i];
            }

            T s_norm = detail::norm2(h_s.data(), n);
            result.relative_residual = s_norm / b_norm;
            result.residual_history.push_back(s_norm);

            if (this->config_.verbose) {
                std::printf("BiCGSTAB iter %d: residual = %.6e, relative = %.6e\n",
                           iter, s_norm, result.relative_residual);
            }

            if (result.relative_residual < this->config_.relative_tolerance) {
                for (int i = 0; i < n; ++i) {
                    h_x[i] += alpha * h_p[i];
                }
                result.converged = true;
                result.iterations = iter + 1;
                result.residual_norm = s_norm;
                result.error_code = SolverError::SUCCESS;
                for (int i = 0; i < n; ++i) {
                    x[i] = h_x[i];
                }
                return result;
            }

            d_s.copy_from(h_s.data(), n);
            d_temp.fill(T{0});
            spmv(A, d_s.data(), d_temp.data());
            d_temp.copy_to(h_t.data(), n);

            T t_s = detail::dot_product(h_t.data(), h_s.data(), n);
            T t_t = detail::dot_product(h_t.data(), h_t.data(), n);

            if (std::abs(t_t) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T omega = t_s / t_t;

            for (int i = 0; i < n; ++i) {
                h_x[i] += omega * h_s[i];
                h_r[i] = h_s[i] - omega * h_t[i];
            }

            T r_norm = detail::norm2(h_r.data(), n);
            result.relative_residual = r_norm / b_norm;
            result.residual_history.push_back(r_norm);

            if (this->config_.verbose) {
                std::printf("BiCGSTAB iter %d: residual = %.6e, relative = %.6e\n",
                           iter + 1, r_norm, result.relative_residual);
            }

            if (result.relative_residual < this->config_.relative_tolerance) {
                result.converged = true;
                result.iterations = iter + 2;
                result.residual_norm = r_norm;
                result.error_code = SolverError::SUCCESS;
                for (int i = 0; i < n; ++i) {
                    x[i] = h_x[i];
                }
                return result;
            }

            T r_new_r_tilde = detail::dot_product(h_r.data(), h_r_tilde.data(), n);

            if (std::abs(r_r_tilde) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T beta = (r_new_r_tilde / r_r_tilde) * (alpha / omega);

            r_r_tilde = r_new_r_tilde;

            for (int i = 0; i < n; ++i) {
                h_p[i] = h_r[i] + beta * (h_p[i] - omega * h_t[i]);
            }
        }

        result.iterations = this->config_.max_iterations;
        result.residual_norm = detail::norm2(h_r.data(), n);
        result.error_code = SolverError::MAX_ITERATIONS;
        for (int i = 0; i < n; ++i) {
            x[i] = h_x[i];
        }
        return result;
    }
};

}

}

#endif
