#pragma once

#include <cstddef>
#include <cstdint>

enum class ReduceOp { SUM, MAX, MIN };

template<typename T>
T reduceSum(const T* d_input, size_t size);

template<typename T>
T reduceMax(const T* d_input, size_t size);

template<typename T>
T reduceMin(const T* d_input, size_t size);

template<typename T>
T reduceSumOptimized(const T* d_input, size_t size);
