#pragma once

#include <cstddef>
#include <cstdint>

template<typename T>
void exclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void inclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void exclusiveScanOptimized(const T* d_input, T* d_output, size_t size);
