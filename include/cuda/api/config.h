#pragma once

#include <cstddef>

namespace cuda::api {

struct ReduceConfig {
    size_t block_size{256};
    size_t min_grid_size{0};
    bool use_optimized{false};

    static ReduceConfig default_config() {
        return ReduceConfig{};
    }

    static ReduceConfig optimized_config() {
        return ReduceConfig{.block_size = 256, .use_optimized = true};
    }
};

struct ScanConfig {
    size_t block_size{256};
    bool use_optimized{true};
    bool inclusive{false};

    static ScanConfig default_config() {
        return ScanConfig{};
    }

    static ScanConfig exclusive_config() {
        return ScanConfig{.block_size = 256, .use_optimized = true, .inclusive = false};
    }

    static ScanConfig inclusive_config() {
        return ScanConfig{.block_size = 256, .use_optimized = true, .inclusive = true};
    }
};

struct SortConfig {
    size_t block_size{256};
    bool use_cub{true};

    static SortConfig default_config() {
        return SortConfig{};
    }
};

struct MatrixMultConfig {
    size_t block_size{16};
    size_t tile_size{16};
    bool use_tiled{true};

    static MatrixMultConfig default_config() {
        return MatrixMultConfig{};
    }

    static MatrixMultConfig naive_config() {
        return MatrixMultConfig{.block_size = 16, .tile_size = 16, .use_tiled = false};
    }
};

} // namespace cuda::api
