#pragma once

#include <cstddef>

template <typename T>
void transposeMatrix(const T* d_input, T* d_output, int rows, int cols);

template <typename T>
void transposeMatrixTiled(const T* d_input, T* d_output, int rows, int cols);

template <typename T>
void matrixElementwiseAdd(const T* d_a, const T* d_b, T* d_c, int rows, int cols);

template <typename T>
void matrixElementwiseMultiply(const T* d_a, const T* d_b, T* d_c, int rows, int cols);

template <typename T>
void matrixScale(T* d_matrix, T scalar, int size);
