#pragma once

#include "cuda/memory/buffer.h"

#include <stdexcept>

namespace cuda::algo {

/**
 * @brief Elementwise addition of two row-major matrices: c = a + b
 *
 * The inputs are taken by const reference and the output by reference.
 *
 * The previous signature took all three BY VALUE, but memory::Buffer deletes its
 * copy constructor and copy assignment (it owns device memory), so no caller
 * could compile: passing a Buffer by value requires a copy. Only a std::move at
 * the call site made it build, and that destroyed the caller's inputs to
 * perform a read-only operation.
 *
 * @param a Left input, numRows * numCols floats
 * @param b Right input, numRows * numCols floats
 * @param c Output, numRows * numCols floats; overwritten
 * @param numRows Row count, must be positive
 * @param numCols Column count, must be positive
 *
 * @throws std::invalid_argument if a dimension is not positive
 * @throws std::invalid_argument if an input or output is smaller than needed
 */
void matrixAdd(const memory::Buffer<float>& a,
               const memory::Buffer<float>& b,
               memory::Buffer<float>& c,
               int numRows, int numCols);

}
