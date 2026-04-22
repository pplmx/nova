#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @brief Exclusive prefix sum (scan) operation on GPU
 * @details All scan functions support sizes up to 1024 elements only.
 *          Sizes greater than 1024 will cause the program to exit with an error.
 */

template<typename T>
void exclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void inclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void exclusiveScanOptimized(const T* d_input, T* d_output, size_t size);
