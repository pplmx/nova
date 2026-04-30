#include "cuda/observability/occupancy_analyzer.h"

#include <algorithm>
#include <sstream>

namespace cuda::observability {

OccupancyAnalyzer::OccupancyAnalyzer(int device)
    : device_(device), sm_count_(0), max_threads_per_sm_(0),
      max_blocks_per_sm_(0), warp_size_(32) {

    int old_device;
    cudaGetDevice(&old_device);
    cudaSetDevice(device_);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_) == cudaSuccess) {
        sm_count_ = prop.multiProcessorCount;
        max_threads_per_sm_ = prop.maxThreadsPerMultiProcessor;
        max_blocks_per_sm_ = prop.maxBlocksPerMultiProcessor;
        warp_size_ = prop.warpSize;
    }

    cudaSetDevice(old_device);
}

OccupancyAnalysis OccupancyAnalyzer::analyze(const void* kernel_func,
                                             int block_size,
                                             size_t dynamic_smem_bytes) {
    OccupancyAnalysis analysis{};
    analysis.theoretical_occupancy = 0.0;
    analysis.achieved_occupancy = 0.0;
    analysis.active_blocks_per_sm = 0;
    analysis.max_blocks_per_sm = 0;
    analysis.max_threads_per_block = block_size;
    analysis.min_blocks_for_full_occupancy = 0;

    if (!kernel_func) {
        return analysis;
    }

    int old_device;
    cudaGetDevice(&old_device);
    cudaSetDevice(device_);

    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &analysis.active_blocks_per_sm,
        kernel_func,
        block_size,
        dynamic_smem_bytes);

    if (err == cudaSuccess) {
        analysis.max_blocks_per_sm = analysis.active_blocks_per_sm;
        int threads_per_block = block_size;
        int max_threads_for_occupancy = threads_per_block * analysis.active_blocks_per_sm;
        analysis.theoretical_occupancy =
            static_cast<double>(max_threads_for_occupancy) / max_threads_per_sm_;

        int min_blocks = (max_threads_per_sm_ + threads_per_block - 1) / threads_per_block;
        analysis.min_blocks_for_full_occupancy = min_blocks;
    }

    cudaSetDevice(old_device);
    return analysis;
}

OccupancyRecommendation OccupancyAnalyzer::recommend(const void* kernel_func,
                                                     size_t dynamic_smem_bytes,
                                                     int preferred_block_size) {
    OccupancyRecommendation rec{};
    rec.recommended_block_size = 256;
    rec.recommended_grid_size = 0;
    rec.expected_occupancy = 0.0;
    rec.registers_per_thread = 0;
    rec.shared_memory_bytes = dynamic_smem_bytes;

    if (!kernel_func) {
        return rec;
    }

    int old_device;
    cudaGetDevice(&old_device);
    cudaSetDevice(device_);

    int block_size_hint = preferred_block_size > 0 ? preferred_block_size : 0;
    int num_blocks = 0;
    size_t dynamic_smem = dynamic_smem_bytes;

    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &num_blocks,
        &rec.recommended_block_size,
        kernel_func,
        dynamic_smem,
        block_size_hint);

    if (err == cudaSuccess) {
        rec.recommended_grid_size = num_blocks * sm_count_;

        int active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            kernel_func,
            rec.recommended_block_size,
            dynamic_smem);

        int max_threads = rec.recommended_block_size * active_blocks;
        rec.expected_occupancy =
            static_cast<double>(max_threads) / max_threads_per_sm_;
    }

    cudaSetDevice(old_device);
    return rec;
}

std::vector<OccupancyAnalysis> OccupancyAnalyzer::analyze_range(
    const void* kernel_func,
    int min_block_size,
    int max_block_size,
    size_t dynamic_smem_bytes) {

    std::vector<OccupancyAnalysis> results;

    for (int block_size = min_block_size; block_size <= max_block_size;
         block_size += warp_size_) {
        results.push_back(analyze(kernel_func, block_size, dynamic_smem_bytes));
    }

    return results;
}

std::string BlockSizeFeedback::to_string() const {
    std::ostringstream oss;
    oss << "Block size: " << block_size
        << ", Occupancy: " << (occupancy * 100.0) << "%";
    if (is_optimal) {
        oss << " [OPTIMAL]";
    }
    return oss.str();
}

std::vector<BlockSizeFeedback> analyze_block_sizes(const void* kernel_func,
                                                   size_t dynamic_smem_bytes,
                                                   int device) {
    std::vector<BlockSizeFeedback> results;

    OccupancyAnalyzer analyzer(device);

    double max_occupancy = 0.0;
    int optimal_block_size = 256;

    for (int block_size = 32; block_size <= 1024; block_size += 32) {
        auto analysis = analyzer.analyze(kernel_func, block_size, dynamic_smem_bytes);

        BlockSizeFeedback fb;
        fb.block_size = block_size;
        fb.occupancy = analysis.theoretical_occupancy;
        fb.is_optimal = false;

        if (fb.occupancy > max_occupancy) {
            max_occupancy = fb.occupancy;
            optimal_block_size = block_size;
        }

        results.push_back(fb);
    }

    for (auto& fb : results) {
        if (fb.block_size == optimal_block_size) {
            fb.is_optimal = true;
        }
    }

    return results;
}

}  // namespace cuda::observability
