#ifndef NOVA_CUDA_SPARSE_KRYLOV_HPP
#define NOVA_CUDA_SPARSE_KRYLOV_HPP

#include "sparse_matrix.hpp"
#include "sparse_ops.hpp"
#include <vector>
#include <cmath>
#include <limits>

namespace nova {
namespace sparse {

enum class SolverError {
    SUCCESS = 0,
    MAX_ITERATIONS,
    BREAKDOWN,
    INVALID_MATRIX,
    CONVERGENCE_FAILURE
};

template<typename T>
struct SolverResult {
    bool converged = false;
    int iterations = 0;
    T residual_norm = T{0};
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

    virtual SolverResult<T> solve(const SparseMatrixCSR<T>& A, const T* b, T* x) = 0;

protected:
    SolverConfig<T> config_;
};

template<typename T>
class ConjugateGradient : public KrylovSolver<T> {
public:
    using KrylovSolver<T>::KrylovSolver;

    SolverResult<T> solve(const SparseMatrixCSR<T>& A, const T* b, T* x) override {
        SolverResult<T> result;
        const int n = A.num_rows();

        if (A.num_rows() != A.num_cols()) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        std::vector<T> r(n), p(n), Ap(n);

        sparse_mv(A, x, Ap.begin());
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - Ap[i];
        }

        T b_norm = detail::norm2(b, n);
        if (b_norm < std::numeric_limits<T>::epsilon()) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        detail::copy(r.data(), p.data(), n);
        T r_dot_old = detail::dot_product(r.data(), r.data(), n);
        T residual_init = std::sqrt(r_dot_old);

        result.residual_history.reserve(this->config_.max_iterations);

        for (int iter = 0; iter < this->config_.max_iterations; ++iter) {
            sparse_mv(A, p.data(), Ap.begin());

            T p_Ap = detail::dot_product(p.data(), Ap.data(), n);
            if (std::abs(p_Ap) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T alpha = r_dot_old / p_Ap;

            detail::axpy(alpha, p.data(), x, n);
            detail::axpy(-alpha, Ap.data(), r.data(), n);

            T r_dot_new = detail::dot_product(r.data(), r.data(), n);
            T residual = std::sqrt(r_dot_new);

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
                return result;
            }

            T beta = r_dot_new / r_dot_old;
            detail::scale(p.data(), beta, n);
            detail::axpy(T{1}, r.data(), p.data(), n);

            r_dot_old = r_dot_new;
        }

        result.iterations = this->config_.max_iterations;
        result.residual_norm = detail::norm2(r.data(), n);
        result.error_code = SolverError::MAX_ITERATIONS;
        return result;
    }
};

template<typename T>
class GMRES : public KrylovSolver<T> {
public:
    GMRES(const SolverConfig<T>& config = {}, int restart = 50)
        : KrylovSolver<T>(config), restart_(restart) {}

    SolverResult<T> solve(const SparseMatrixCSR<T>& A, const T* b, T* x) override {
        SolverResult<T> result;
        const int n = A.num_rows();

        if (A.num_rows() != A.num_cols()) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        T b_norm = detail::norm2(b, n);
        if (b_norm < std::numeric_limits<T>::epsilon()) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        std::vector<T> r(n), v(n);
        std::vector<T> w(n);

        std::vector<T> cos_sin(restart_);
        std::vector<T> s(restart_ + 1), cs(restart_), sn(restart_);

        sparse_mv(A, x, v.begin());
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - v[i];
        }

        T beta = detail::norm2(r.data(), n);

        int total_iterations = 0;

        for (int outer = 0; outer < this->config_.max_iterations / restart_; ++outer) {
            T* V = new T[(restart_ + 1) * n];
            T* H = new T[(restart_ + 1) * restart_];

            for (int i = 0; i < (restart_ + 1) * restart_; ++i) {
                H[i] = T{0};
            }

            for (int i = 0; i < (restart_ + 1) * n; ++i) {
                V[i] = T{0};
            }

            for (int i = 0; i < n; ++i) {
                V[i] = r[i] / beta;
            }

            s[0] = beta;
            for (int i = 1; i <= restart_; ++i) {
                s[i] = T{0};
            }

            for (int j = 0; j < restart_ && total_iterations < this->config_.max_iterations; ++j) {
                for (int i = 0; i < n; ++i) {
                    v[i] = V[j * n + i];
                }

                sparse_mv(A, v.begin(), w.begin());

                T h_ij = T{0};
                for (int i = 0; i < n; ++i) {
                    h_ij += w[i] * V[j * n + i];
                    H[j * restart_ + j] = h_ij;
                }

                for (int i = 0; i < n; ++i) {
                    w[i] -= h_ij * V[j * n + i];
                }

                for (int k = 0; k < j; ++k) {
                    T h_kj = H[k * restart_ + j];
                    for (int i = 0; i < n; ++i) {
                        w[i] -= h_kj * V[k * n + i];
                    }
                    H[k * restart_ + j] = h_ij - h_kj * (H[k * restart_ + j] / h_ij) * h_ij;
                }

                h_ij = detail::norm2(w.data(), n);

                bool breakdown = h_ij < std::numeric_limits<T>::epsilon();
                if (breakdown) {
                    h_ij = T{1};
                    result.error_code = SolverError::BREAKDOWN;
                }

                for (int i = 0; i < n; ++i) {
                    V[(j + 1) * n + i] = w[i] / h_ij;
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

                T c = s[j] / std::sqrt(s[j] * s[j] + h_ij * h_ij);
                T s_k = h_ij / std::sqrt(s[j] * s[j] + h_ij * h_ij);
                cs[j] = c;
                sn[j] = s_k;
                H[j * restart_ + j] = c * h_ij;
                s[j] = c * s[j];
                s[j + 1] = -s_k * s[j + 1];

                T residual = std::abs(s[j + 1]);
                result.residual_history.push_back(residual);
                result.relative_residual = residual / b_norm;

                if (this->config_.verbose) {
                    std::printf("GMRES iter %d: residual = %.6e, relative = %.6e\n",
                               total_iterations, residual, result.relative_residual);
                }

                ++total_iterations;

                if (result.relative_residual < this->config_.relative_tolerance) {
                    std::vector<T> y(restart_, T{0});
                    for (int k = j; k >= 0; --k) {
                        y[k] = s[k];
                        for (int i = k + 1; i <= j; ++i) {
                            y[k] -= H[k * restart_ + i] * y[i];
                        }
                        y[k] /= H[k * restart_ + k];
                    }

                    for (int k = 0; k <= j; ++k) {
                        for (int i = 0; i < n; ++i) {
                            x[i] += y[k] * V[k * n + i];
                        }
                    }

                    result.converged = true;
                    result.iterations = total_iterations;
                    result.residual_norm = residual;
                    result.error_code = SolverError::SUCCESS;

                    delete[] V;
                    delete[] H;
                    return result;
                }
            }

            std::vector<T> y(restart_, T{0});
            for (int k = restart_ - 1; k >= 0; --k) {
                y[k] = s[k];
                for (int i = k + 1; i < restart_; ++i) {
                    y[k] -= H[k * restart_ + i] * y[i];
                }
                if (std::abs(H[k * restart_ + k]) > std::numeric_limits<T>::epsilon()) {
                    y[k] /= H[k * restart_ + k];
                }
            }

            for (int k = 0; k < restart_; ++k) {
                for (int i = 0; i < n; ++i) {
                    x[i] += y[k] * V[k * n + i];
                }
            }

            sparse_mv(A, x, v.begin());
            for (int i = 0; i < n; ++i) {
                r[i] = b[i] - v[i];
            }
            beta = detail::norm2(r.data(), n);

            result.relative_residual = beta / b_norm;

            if (this->config_.verbose) {
                std::printf("GMRES restart %d: residual = %.6e, relative = %.6e\n",
                           outer, beta, result.relative_residual);
            }

            if (result.relative_residual < this->config_.relative_tolerance) {
                result.converged = true;
                result.iterations = total_iterations;
                result.residual_norm = beta;
                result.error_code = SolverError::SUCCESS;

                delete[] V;
                delete[] H;
                return result;
            }

            delete[] V;
            delete[] H;

            if (total_iterations >= this->config_.max_iterations) {
                break;
            }
        }

        result.iterations = total_iterations;
        result.residual_norm = beta;
        result.error_code = SolverError::MAX_ITERATIONS;
        return result;
    }

private:
    int restart_;
};

template<typename T>
class BiCGSTAB : public KrylovSolver<T> {
public:
    using KrylovSolver<T>::KrylovSolver;

    SolverResult<T> solve(const SparseMatrixCSR<T>& A, const T* b, T* x) override {
        SolverResult<T> result;
        const int n = A.num_rows();

        if (A.num_rows() != A.num_cols()) {
            result.error_code = SolverError::INVALID_MATRIX;
            return result;
        }

        std::vector<T> r(n), r_tilde(n), p(n), p_hat(n), s(n), t(n);

        sparse_mv(A, x, p.begin());
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - p[i];
        }

        detail::copy(r.data(), r_tilde.data(), n);

        T b_norm = detail::norm2(b, n);
        if (b_norm < std::numeric_limits<T>::epsilon()) {
            result.converged = true;
            result.iterations = 0;
            result.error_code = SolverError::SUCCESS;
            return result;
        }

        detail::copy(r.data(), p.data(), n);

        T r_r_tilde = detail::dot_product(r.data(), r_tilde.data(), n);

        result.residual_history.reserve(this->config_.max_iterations);

        for (int iter = 0; iter < this->config_.max_iterations; ++iter) {
            detail::copy(p.data(), p_hat.data(), n);

            sparse_mv(A, p_hat.data(), s.begin());

            T s_r_tilde = detail::dot_product(s.data(), r_tilde.data(), n);
            if (std::abs(s_r_tilde) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T alpha = r_r_tilde / s_r_tilde;

            for (int i = 0; i < n; ++i) {
                s[i] = r[i] - alpha * s[i];
            }

            for (int i = 0; i < n; ++i) {
                x[i] += alpha * p_hat[i];
            }

            T s_norm = detail::norm2(s.data(), n);
            result.relative_residual = s_norm / b_norm;

            result.residual_history.push_back(s_norm);

            if (this->config_.verbose) {
                std::printf("BiCGSTAB iter %d: residual = %.6e, relative = %.6e\n",
                           iter, s_norm, result.relative_residual);
            }

            if (result.relative_residual < this->config_.relative_tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                result.residual_norm = s_norm;
                result.error_code = SolverError::SUCCESS;
                return result;
            }

            sparse_mv(A, s.data(), t.begin());

            T t_s = detail::dot_product(t.data(), s.data(), n);
            T t_t = detail::dot_product(t.data(), t.data(), n);

            if (std::abs(t_t) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T omega = t_s / t_t;

            for (int i = 0; i < n; ++i) {
                x[i] += omega * s[i];
            }

            for (int i = 0; i < n; ++i) {
                r[i] = s[i] - omega * t[i];
            }

            T r_norm = detail::norm2(r.data(), n);
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
                return result;
            }

            T r_new_r_tilde = detail::dot_product(r.data(), r_tilde.data(), n);

            if (std::abs(r_r_tilde) < std::numeric_limits<T>::epsilon()) {
                result.error_code = SolverError::BREAKDOWN;
                break;
            }

            T beta = (r_new_r_tilde / r_r_tilde) * (alpha / omega);

            r_r_tilde = r_new_r_tilde;

            for (int i = 0; i < n; ++i) {
                p[i] = r[i] + beta * (p[i] - omega * s[i]);
            }
        }

        result.iterations = this->config_.max_iterations;
        result.residual_norm = detail::norm2(r.data(), n);
        result.error_code = SolverError::MAX_ITERATIONS;
        return result;
    }
};

}
}

#endif
