#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

/**
 * @brief Exclusive prefix sum (scan) operation on GPU
 * @details All scan functions support sizes up to 1024 elements only.
 * @throws std::invalid_argument if size exceeds 1024
 */

constexpr size_t MAX_SCAN_SIZE = 1024;

class ScanSizeException : public std::invalid_argument {
public:
    explicit ScanSizeException(size_t size, size_t max_size)
        : std::invalid_argument("Scan size " + std::to_string(size) +
                                " exceeds maximum supported size " + std::to_string(max_size)) {}
};

template<typename T>
void exclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void inclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void exclusiveScanOptimized(const T* d_input, T* d_output, size_t size);
