#pragma once

#include "cuda/memory/buffer.h"
#include <cstddef>
#include <stdexcept>
#include <string>

constexpr size_t MAX_SCAN_SIZE = 1024;

class ScanSizeException : public std::invalid_argument {
public:
    explicit ScanSizeException(size_t size, size_t max_size)
        : std::invalid_argument("Scan size " + std::to_string(size) + " exceeds maximum supported size " + std::to_string(max_size)) {}
};

namespace cuda::algo {

    template <typename T>
    void exclusiveScan(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size);

    template <typename T>
    void inclusiveScan(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size);

    template <typename T>
    void exclusiveScanOptimized(const memory::Buffer<T>& input, memory::Buffer<T>& output, size_t size);

}  // namespace cuda::algo
