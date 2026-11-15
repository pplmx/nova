/**
 * @file solver_workspace.hpp
 * @brief Workspace management for iterative sparse solvers
 * @defgroup solver_workspace Solver Workspace
 * @ingroup sparse
 *
 * Manages temporary vectors used by iterative solvers (CG, GMRES, BiCGSTAB).
 * Pre-allocates all required workspace to avoid allocation overhead during solve.
 *
 * @see krylov.hpp For solver implementations
 */

#ifndef NOVA_CUDA_SPARSE_SOLVER_WORKSPACE_HPP
#define NOVA_CUDA_SPARSE_SOLVER_WORKSPACE_HPP

#include "solver_errors.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace nova {
namespace sparse {

/**
 * @brief Workspace for iterative solvers
 * @class SolverWorkspace
 * @tparam T Element type
 * @ingroup solver_workspace
 *
 * Pre-allocates temporary vectors used by Krylov subspace methods.
 */
template<typename T>
class SolverWorkspace {
public:
    /** @brief Default constructor creates empty workspace */
    SolverWorkspace() : size_(0) {}

    /**
     * @brief Construct and allocate workspace
     * @param size Problem size (number of unknowns)
     */
    explicit SolverWorkspace(int size) : size_(0) {
        resize(size);
    }

    /**
     * @brief Resize workspace for new problem size
     * @param size Problem size
     */
    void resize(int size) {
        if (size == size_) return;

        size_ = size;
        r_.resize(size);
        p_.resize(size);
        Ap_.resize(size);
        r_tilde_.resize(size);
        p_hat_.resize(size);
        s_.resize(size);
        t_.resize(size);
    }

    void reset() {
        std::fill(r_.begin(), r_.end(), T{0});
        std::fill(p_.begin(), p_.end(), T{0});
        std::fill(Ap_.begin(), Ap_.end(), T{0});
        std::fill(r_tilde_.begin(), r_tilde_.end(), T{0});
        std::fill(p_hat_.begin(), p_hat_.end(), T{0});
        std::fill(s_.begin(), s_.end(), T{0});
        std::fill(t_.begin(), t_.end(), T{0});
    }

    int size() const { return size_; }

    std::vector<T>& r() { return r_; }
    std::vector<T>& p() { return p_; }
    std::vector<T>& Ap() { return Ap_; }
    std::vector<T>& r_tilde() { return r_tilde_; }
    std::vector<T>& p_hat() { return p_hat_; }
    std::vector<T>& s() { return s_; }
    std::vector<T>& t() { return t_; }

    const std::vector<T>& r() const { return r_; }
    const std::vector<T>& p() const { return p_; }
    const std::vector<T>& Ap() const { return Ap_; }
    const std::vector<T>& r_tilde() const { return r_tilde_; }
    const std::vector<T>& p_hat() const { return p_hat_; }
    const std::vector<T>& s() const { return s_; }
    const std::vector<T>& t() const { return t_; }

private:
    int size_;
    std::vector<T> r_, p_, Ap_;
    std::vector<T> r_tilde_, p_hat_, s_, t_;
};

struct SolverDiagnostics {
    double setup_time_ms = 0.0;
    double solve_time_ms = 0.0;
    double total_time_ms = 0.0;
    std::vector<double> convergence_history;

    double convergence_rate = 0.0;
    double average_convergence_rate() const {
        if (convergence_history.size() < 2) return 0.0;

        double product = 1.0;
        for (size_t i = 1; i < convergence_history.size(); ++i) {
            if (convergence_history[i-1] > 0) {
                product *= convergence_history[i] / convergence_history[i-1];
            }
        }
        return std::pow(product, 1.0 / static_cast<double>(convergence_history.size() - 1));
    }

    void compute_convergence_rate(const std::vector<double>& residual_history) {
        convergence_history = residual_history;
        convergence_rate = average_convergence_rate();
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Solver Diagnostics:\n";
        oss << "  Setup time: " << setup_time_ms << " ms\n";
        oss << "  Solve time: " << solve_time_ms << " ms\n";
        oss << "  Total time: " << total_time_ms << " ms\n";
        oss << "  Convergence rate: " << convergence_rate << "\n";
        oss << "  Iterations: " << convergence_history.size();
        return oss.str();
    }
};

template<typename T>
struct TimedSolverResult {
    bool converged = false;
    int iterations = 0;
    T residual_norm = T{0};
    T relative_residual = T{0};
    SolverError error_code = SolverError::SUCCESS;
    std::vector<T> residual_history;
    SolverDiagnostics diagnostics;
};

}
}

#endif
