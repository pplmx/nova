#pragma once

/**
 * @file kernel_launcher.h
 * @brief Kernel launch abstraction with builder pattern
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

#include "cuda/device/error.h"

namespace cuda::detail {

/**
 * @class KernelLauncher
 * @brief Builder-pattern kernel launcher for CUDA kernels.
 *
 * Provides a fluent interface for configuring and launching CUDA kernels.
 * All configuration methods return a reference to enable chaining.
 *
 * @example
 * @code
 * KernelLauncher launcher;
 * launcher.block({256, 1, 1})
 *       .grid(calc_grid_1d(n))
 *       .shared(256 * sizeof(float))
 *       .launch(my_kernel, data, size);
 * launcher.synchronize();
 * @endcode
 */
class KernelLauncher {
public:
    /**
     * @brief Default constructor with reasonable defaults
     */
    KernelLauncher() = default;

    /**
     * @brief Sets the grid dimensions
     * @param g Grid configuration (blocks in x, y, z)
     * @return Reference to this launcher for chaining
     */
    KernelLauncher& grid(dim3 g) & {
        grid_ = g;
        return *this;
    }

    /**
     * @brief Sets the block dimensions
     * @param b Block configuration (threads in x, y, z)
     * @return Reference to this launcher for chaining
     */
    KernelLauncher& block(dim3 b) & {
        block_ = b;
        return *this;
    }

    /**
     * @brief Sets the shared memory size in bytes
     * @param s Shared memory per block in bytes
     * @return Reference to this launcher for chaining
     */
    KernelLauncher& shared(size_t s) & {
        shared_ = s;
        return *this;
    }

    /**
     * @brief Sets the CUDA stream for the launch
     * @param s CUDA stream handle (nullptr for default stream)
     * @return Reference to this launcher for chaining
     */
    KernelLauncher& stream(cudaStream_t s) & {
        stream_ = s;
        return *this;
    }

    /**
     * @brief Launches the kernel with current configuration
     * @tparam Kernel Kernel function type
     * @tparam Args Argument types
     * @param kernel Pointer to the kernel function
     * @param args Arguments to pass to the kernel
     * @throws CudaException if launch or configuration validation fails
     */
    template <typename Kernel, typename... Args>
    void launch(Kernel* kernel, Args&&... args) {
        void* ptrs[] = {const_cast<std::remove_cv_t<std::remove_reference_t<Args>>*>(&args)...};
        CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid_, block_, ptrs, shared_, stream_));
        CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief Synchronizes with the kernel completion
     * @throws CudaException if synchronization fails
     */
    void synchronize() const {
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    /**
     * @brief Gets current grid configuration
     */
    [[nodiscard]] dim3 get_grid() const { return grid_; }

    /**
     * @brief Gets current block configuration
     */
    [[nodiscard]] dim3 get_block() const { return block_; }

    /**
     * @brief Gets shared memory size
     */
    [[nodiscard]] size_t get_shared() const { return shared_; }

    /**
     * @brief Gets CUDA stream
     */
    [[nodiscard]] cudaStream_t get_stream() const { return stream_; }

private:
    dim3 grid_{1, 1, 1};
    dim3 block_{1, 1, 1};
    size_t shared_{0};
    cudaStream_t stream_{nullptr};
};

/**
 * @brief Calculates optimal grid size for 1D kernel execution
 * @param n Total number of elements to process
 * @param block Block size (default: 256 threads)
 * @return Grid configuration with sufficient blocks
 */
[[nodiscard]] constexpr inline dim3 calc_grid_1d(size_t n, dim3 block = {256, 1, 1}) {
    return dim3{static_cast<unsigned int>((n + block.x - 1) / block.x), 1, 1};
}

/**
 * @brief Calculates optimal grid size for 2D kernel execution
 * @param w Width (x dimension)
 * @param h Height (y dimension)
 * @param block Block configuration (default: 16x16)
 * @return Grid configuration with sufficient blocks
 */
[[nodiscard]] constexpr inline dim3 calc_grid_2d(size_t w, size_t h, dim3 block = {16, 16, 1}) {
    return dim3{static_cast<unsigned int>((w + block.x - 1) / block.x), static_cast<unsigned int>((h + block.y - 1) / block.y), 1};
}

/**
 * @brief Calculates optimal grid size for 3D kernel execution
 * @param x X dimension size
 * @param y Y dimension size
 * @param z Z dimension size
 * @param block Block configuration (default: 8x8x8)
 * @return Grid configuration with sufficient blocks
 */
[[nodiscard]] constexpr inline dim3 calc_grid_3d(size_t x, size_t y, size_t z, dim3 block = {8, 8, 8}) {
    return dim3{
        static_cast<unsigned int>((x + block.x - 1) / block.x), static_cast<unsigned int>((y + block.y - 1) / block.y), static_cast<unsigned int>((z + block.z - 1) / block.z)};
}

}  // namespace cuda::detail
