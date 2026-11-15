/**
 * @file sparse_concepts.hpp
 * @brief Constraints for the scalar type used by the sparse templates
 * @defgroup sparse_concepts Sparse Concepts
 * @ingroup sparse
 *
 * Every sparse template was written as `template<typename T>` with no
 * constraint, while the implementation assumes T supports arithmetic
 * (std::abs, operator*, comparison against 0) and conversion to double. Passing
 * anything else produced a cascade of errors from inside the template body —
 * SparseMatrix<std::string> gave six unrelated complaints about narrowing and
 * missing operators instead of one statement that the scalar type is wrong.
 *
 * These concepts make the requirement part of the interface, so the diagnostic
 * names the constraint that failed and the offending instantiation.
 *
 * @note Host-only (C++23). The device-reachable sparse headers (matrix.hpp,
 *       cusparse_context.hpp) stay unconstrained because nvcc cannot compile
 *       concepts.
 */

#ifndef NOVA_CUDA_SPARSE_CONCEPTS_HPP
#define NOVA_CUDA_SPARSE_CONCEPTS_HPP

#include <cuda/cpp_level.hpp>

#include <concepts>
#include <type_traits>

namespace nova::sparse {

NOVA_REQUIRE_CPP23_HOST("sparse_concepts.hpp")


/**
 * @brief Scalar type an iterative solver or preconditioner can work with
 *
 * Floating point covers every instantiation in the tree (float and double) and
 * is what the algorithms need: a triangular solve and a residual norm are
 * floating-point operations. Complex is not supported by the kernels, so it is
 * deliberately excluded rather than silently accepted.
 */
template<typename T>
concept SparseScalar = std::floating_point<T>;

}  // namespace nova::sparse

#endif  // NOVA_CUDA_SPARSE_CONCEPTS_HPP
