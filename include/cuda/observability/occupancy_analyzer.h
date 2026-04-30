#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

namespace cuda::observability {

struct OccupancyRecommendation {
    int recommended_block_size;
    int recommended_grid_size;
    double expected_occupancy;
    size_t registers_per_thread;
    size_t shared_memory_bytes;
};

struct OccupancyAnalysis {
    double theoretical_occupancy;
    double achieved_occupancy;
    int active_blocks_per_sm;
    int max_blocks_per_sm;
    int max_threads_per_block;
    int min_blocks_for_full_occupancy;
};

class OccupancyAnalyzer {
public:
    OccupancyAnalyzer(int device = 0);

    OccupancyAnalysis analyze(const void* kernel_func,
                              int block_size,
                              size_t dynamic_smem_bytes = 0);

    OccupancyRecommendation recommend(const void* kernel_func,
                                      size_t dynamic_smem_bytes = 0,
                                      int preferred_block_size = 0);

    std::vector<OccupancyAnalysis> analyze_range(const void* kernel_func,
                                                 int min_block_size,
                                                 int max_block_size,
                                                 size_t dynamic_smem_bytes = 0);

    int device() const { return device_; }
    int sm_count() const { return sm_count_; }
    int max_threads_per_sm() const { return max_threads_per_sm_; }

private:
    int device_;
    int sm_count_;
    int max_threads_per_sm_;
    int max_blocks_per_sm_;
    int warp_size_;
};

struct BlockSizeFeedback {
    int block_size;
    double occupancy;
    bool is_optimal;

    std::string to_string() const;
};

std::vector<BlockSizeFeedback> analyze_block_sizes(const void* kernel_func,
                                                   size_t dynamic_smem_bytes = 0,
                                                   int device = 0);

}  // namespace cuda::observability
