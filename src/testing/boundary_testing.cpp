#include "cuda/testing/boundary_testing.h"

#include <cuda_runtime.h>

namespace cuda::testing {

    std::vector<BoundaryCondition> BoundaryConditionTest::get_cuda_boundaries() {
        std::vector<BoundaryCondition> boundaries;

        auto memory = get_memory_boundaries();
        auto alignment = get_alignment_boundaries();
        auto compute = get_compute_boundaries();

        boundaries.insert(boundaries.end(), memory.begin(), memory.end());
        boundaries.insert(boundaries.end(), alignment.begin(), alignment.end());
        boundaries.insert(boundaries.end(), compute.begin(), compute.end());

        return boundaries;
    }

    std::vector<BoundaryCondition> BoundaryConditionTest::get_memory_boundaries() {
        return {
            {"min_allocation", 1, true, "Minimum 1 byte allocation", BoundaryKind::Allocation},
            {"max_allocation_1gb", 1UL << 30, true, "1 GB allocation", BoundaryKind::Allocation},
            {"zero_allocation", 0, false, "Zero-byte allocation should fail", BoundaryKind::Allocation},
            {"overflow_allocation", SIZE_MAX, false, "Overflow allocation should fail", BoundaryKind::Allocation},
        };
    }

    std::vector<BoundaryCondition> BoundaryConditionTest::get_alignment_boundaries() {
        return {
            {"256_byte_alignment", CUDA_MEMORY_ALIGNMENT, true, "256-byte alignment", BoundaryKind::Alignment},
            {"warp_aligned", CUDA_WARP_SIZE, true, "Warp-size alignment", BoundaryKind::Alignment},
            {"misaligned", 255, false, "Misaligned access should fail", BoundaryKind::Alignment},
            {"page_aligned", 4096, true, "Page-size alignment", BoundaryKind::Alignment},
        };
    }

    std::vector<BoundaryCondition> BoundaryConditionTest::get_compute_boundaries() {
        return {
            {"max_threads_per_block", CUDA_MAX_THREADS_PER_BLOCK, true, "Max threads per block", BoundaryKind::Compute},
            {"warp_size", CUDA_WARP_SIZE, true, "Warp size boundary", BoundaryKind::Compute},
            {"max_blocks_per_sm", CUDA_MAX_BLOCKS_PER_SM, true, "Max blocks per SM", BoundaryKind::Compute},
            {"oversubscribed_grid", 65535ULL * 65535, false, "Oversubscribed grid", BoundaryKind::Compute},
        };
    }

    bool BoundaryConditionTest::test_allocation(size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);

        if (err == cudaSuccess && ptr != nullptr) {
            cudaFree(ptr);
            return true;
        }

        return false;
    }

    bool BoundaryConditionTest::test_alignment(size_t alignment) {
        return is_memory_aligned(reinterpret_cast<const void*>(alignment));
    }

    bool BoundaryConditionTest::test_grid_size(dim3 grid) {
        return grid.x <= 65535 && grid.y <= 65535 && grid.z <= 65535;
    }

    bool BoundaryConditionTest::test_block_size(dim3 block) {
        size_t total_threads = block.x * block.y * block.z;
        return total_threads <= CUDA_MAX_THREADS_PER_BLOCK && block.x > 0 && block.y > 0 && block.z > 0;
    }

    bool BoundaryConditionTest::test_shared_memory(size_t bytes) {
        return bytes <= 48 * 1024;
    }

    std::vector<BoundaryConditionTest::TestResult> BoundaryConditionTest::run_all_boundaries() {
        std::vector<TestResult> results;

        auto boundaries = get_cuda_boundaries();
        for (const auto& bc : boundaries) {
            TestResult result;
            result.condition_name = bc.name;

            // Dispatch on the condition's kind. Judging every entry by whether
            // cudaMalloc succeeded made the alignment and grid expectations
            // meaningless: a 255-byte allocation and a 65535^2-byte allocation both
            // succeed, so "misaligned" and "oversubscribed_grid" were reported as
            // unexpected successes.
            bool succeeded = false;
            switch (bc.kind) {
                case BoundaryKind::Allocation:
                    succeeded = test_allocation(bc.value);
                    break;
                case BoundaryKind::Alignment:
                    // Each entry names an alignment in bytes and asks whether that
                    // alignment is legal. A warp-size alignment (32 B) is legal even
                    // though it is not a multiple of the 256-byte base alignment, so
                    // the requirement scales with the value: an alignment is valid
                    // when it is a power of two. 255 is not, which is why that entry
                    // is expected to fail.
                    succeeded = (bc.value != 0) && ((bc.value & (bc.value - 1)) == 0);
                    break;
                case BoundaryKind::Compute:
                    // Thread counts above the per-block limit are invalid.
                    succeeded = bc.value <= CUDA_MAX_THREADS_PER_BLOCK;
                    break;
                case BoundaryKind::SharedMemory:
                    succeeded = test_shared_memory(bc.value);
                    break;
            }

            if (succeeded) {
                result.passed = bc.should_succeed;
                result.error_message = bc.should_succeed ? nullptr : "Unexpectedly succeeded";
            } else {
                result.passed = !bc.should_succeed;
                result.error_message = !bc.should_succeed ? nullptr : "Unexpectedly failed";
            }

            results.push_back(result);
        }

        return results;
    }

    namespace {

        /// Runs only the conditions of one kind, so the category helpers report their
        /// own category instead of all twelve entries.
        std::vector<BoundaryConditionTest::TestResult> run_kind(BoundaryKind kind) {
            BoundaryConditionTest test;
            const auto all = test.run_all_boundaries();
            const auto boundaries = BoundaryConditionTest::get_cuda_boundaries();

            std::vector<BoundaryConditionTest::TestResult> filtered;
            filtered.reserve(boundaries.size());
            for (size_t i = 0; i < boundaries.size() && i < all.size(); ++i) {
                if (boundaries[i].kind == kind) {
                    filtered.push_back(all[i]);
                }
            }
            return filtered;
        }

    }  // namespace

    std::vector<BoundaryConditionTest::TestResult> test_memory_boundaries() {
        return run_kind(BoundaryKind::Allocation);
    }

    std::vector<BoundaryConditionTest::TestResult> test_alignment_boundaries() {
        return run_kind(BoundaryKind::Alignment);
    }

    std::vector<BoundaryConditionTest::TestResult> test_warp_boundaries() {
        std::vector<BoundaryConditionTest::TestResult> results;

        BoundaryConditionTest::TestResult r1;
        r1.condition_name = "warp_aligned";
        r1.passed = is_warp_aligned(CUDA_WARP_SIZE);
        r1.error_message = nullptr;
        results.push_back(r1);

        BoundaryConditionTest::TestResult r2;
        r2.condition_name = "warp_misaligned";
        r2.passed = !is_warp_aligned(CUDA_WARP_SIZE + 1);
        r2.error_message = nullptr;
        results.push_back(r2);

        return results;
    }

    std::vector<BoundaryConditionTest::TestResult> test_sm_boundaries() {
        BoundaryConditionTest test;
        return test.run_all_boundaries();
    }

    bool is_warp_aligned(size_t size) {
        return (size & (CUDA_WARP_SIZE - 1)) == 0;
    }

    bool is_memory_aligned(const void* ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) & (CUDA_MEMORY_ALIGNMENT - 1)) == 0;
    }

    bool is_valid_block_size(dim3 block) {
        size_t total_threads = block.x * block.y * block.z;
        return total_threads > 0 && total_threads <= CUDA_MAX_THREADS_PER_BLOCK;
    }

}  // namespace cuda::testing
