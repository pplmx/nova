#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace cuda::testing {

struct BoundaryCondition {
    const char* name;
    size_t value;
    bool should_succeed;
    const char* description;
};

class BoundaryConditionTest {
public:
    BoundaryConditionTest() = default;

    static std::vector<BoundaryCondition> get_cuda_boundaries();

    static std::vector<BoundaryCondition> get_memory_boundaries();
    static std::vector<BoundaryCondition> get_alignment_boundaries();
    static std::vector<BoundaryCondition> get_compute_boundaries();

    bool test_allocation(size_t size);
    bool test_alignment(size_t alignment);
    bool test_grid_size(dim3 grid);
    bool test_block_size(dim3 block);
    bool test_shared_memory(size_t bytes);

    struct TestResult {
        const char* condition_name;
        bool passed;
        const char* error_message;
    };

    std::vector<TestResult> run_all_boundaries();
};

std::vector<BoundaryConditionTest::TestResult>
test_memory_boundaries();

std::vector<BoundaryConditionTest::TestResult>
test_alignment_boundaries();

std::vector<BoundaryConditionTest::TestResult>
test_warp_boundaries();

std::vector<BoundaryConditionTest::TestResult>
test_sm_boundaries();

constexpr size_t CUDA_WARP_SIZE = 32;
constexpr size_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
constexpr size_t CUDA_MAX_BLOCKS_PER_SM = 32;
constexpr size_t CUDA_MEMORY_ALIGNMENT = 256;

constexpr size_t CUDA_WARP_MASK = ~(CUDA_WARP_SIZE - 1);
constexpr size_t CUDA_ALIGNMENT_MASK = ~(CUDA_MEMORY_ALIGNMENT - 1);

bool is_warp_aligned(size_t size);
bool is_memory_aligned(const void* ptr);
bool is_valid_block_size(dim3 block);

}  // namespace cuda::testing
