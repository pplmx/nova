/**
 * @file solver_errors.hpp
 * @brief Error codes shared by the iterative sparse solvers
 * @defgroup solver_errors Solver Error Codes
 * @ingroup sparse
 *
 * Lives in its own header so that both the solver implementations (krylov.hpp)
 * and the workspace/diagnostics helpers (solver_workspace.hpp) can name the
 * codes without depending on each other.
 */

#ifndef NOVA_CUDA_SPARSE_SOLVER_ERRORS_HPP
#define NOVA_CUDA_SPARSE_SOLVER_ERRORS_HPP

namespace nova {
namespace sparse {

/**
 * @brief Solver error codes
 * @enum SolverError
 * @ingroup solver_errors
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

}
}

#endif
