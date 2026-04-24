#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "cuda/memory/distributed_pool.h"
#include "cuda/mesh/device_mesh.h"

using cuda::memory::DistributedMemoryPool;

class DistributedMemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<DistributedMemoryPool>();
    }

    void TearDown() override {
        pool_.reset();
    }

    std::unique_ptr<DistributedMemoryPool> pool_;
};

// ============================================================================
// MGPU-09: Per-Device Pool Allocation
// ============================================================================

TEST_F(DistributedMemoryPoolTest, DeviceCountMatchesCUDA) {
    int cuda_device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));

    EXPECT_EQ(pool_->device_count(), cuda_device_count);
}

TEST_F(DistributedMemoryPoolTest, AllocateOnSpecificDevice) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();

    for (int device_id = 0; device_id < mesh.device_count(); ++device_id) {
        size_t bytes = 1024;
        void* ptr = pool_->allocate(bytes, device_id);

        EXPECT_NE(ptr, nullptr);

        // Verify ownership tracking
        auto ownership = pool_->get_ownership(ptr);
        EXPECT_EQ(ownership.ptr, ptr);
        EXPECT_EQ(ownership.owning_device, device_id);
        EXPECT_EQ(ownership.bytes, bytes);

        // Deallocate
        pool_->deallocate(ptr);
    }
}

TEST_F(DistributedMemoryPoolTest, PerDeviceAllocationTracking) {
    // Allocate from multiple devices
    void* ptr0 = pool_->allocate(1024, 0);
    void* ptr1 = pool_->device_count() > 1
                 ? pool_->allocate(2048, 1)
                 : nullptr;

    // Verify metrics for each device
    auto metrics0 = pool_->get_device_metrics(0);
    EXPECT_GT(metrics0.allocated_bytes, 0);

    if (pool_->device_count() > 1) {
        auto metrics1 = pool_->get_device_metrics(1);
        EXPECT_GT(metrics1.allocated_bytes, 0);
    }

    // Cleanup
    pool_->deallocate(ptr0);
    if (ptr1) pool_->deallocate(ptr1);
}

// ============================================================================
// MGPU-10: Auto-Allocation
// ============================================================================

TEST_F(DistributedMemoryPoolTest, AutoAllocateSelectsDevice) {
    if (pool_->device_count() < 2) {
        GTEST_SKIP() << "Requires multiple GPUs for auto-allocation test";
    }

    void* ptr = pool_->allocate_auto(4096);

    // Should have allocated successfully
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(pool_->owns_pointer(ptr));

    // Get best device and verify allocation
    int best_device = pool_->get_best_device();
    auto ownership = pool_->get_ownership(ptr);
    EXPECT_GE(ownership.owning_device, 0);
    EXPECT_LT(ownership.owning_device, pool_->device_count());

    pool_->deallocate(ptr);
}

TEST_F(DistributedMemoryPoolTest, GetBestDeviceReturnsValidId) {
    int best_device = pool_->get_best_device();
    EXPECT_GE(best_device, 0);
    EXPECT_LT(best_device, pool_->device_count());
}

TEST_F(DistributedMemoryPoolTest, AllMetricsAccessible) {
    auto all_metrics = pool_->get_all_metrics();

    EXPECT_EQ(all_metrics.size(), pool_->device_count());

    for (const auto& metrics : all_metrics) {
        EXPECT_GE(metrics.device_id, 0);
        EXPECT_LT(metrics.device_id, pool_->device_count());
        EXPECT_GE(metrics.total_memory, 0);
        EXPECT_GE(metrics.allocated_bytes, 0);
        EXPECT_GE(metrics.free_memory, 0);
    }
}

// ============================================================================
// MGPU-11: Cross-Device Ownership Tracking
// ============================================================================

TEST_F(DistributedMemoryPoolTest, OwnershipTrackingOnAllocate) {
    void* ptr = pool_->allocate(1024, 0);

    EXPECT_TRUE(pool_->owns_pointer(ptr));

    auto ownership = pool_->get_ownership(ptr);
    EXPECT_EQ(ownership.ptr, ptr);
    EXPECT_EQ(ownership.owning_device, 0);
    EXPECT_EQ(ownership.requesting_device, 0);

    pool_->deallocate(ptr);
}

TEST_F(DistributedMemoryPoolTest, OwnershipTrackingAfterDeallocate) {
    void* ptr = pool_->allocate(1024, 0);

    pool_->deallocate(ptr);

    // Should no longer own the pointer
    EXPECT_FALSE(pool_->owns_pointer(ptr));

    // get_ownership should return invalid record
    auto ownership = pool_->get_ownership(ptr);
    EXPECT_EQ(ownership.ptr, nullptr);
}

TEST_F(DistributedMemoryPoolTest, DeallocateFromCorrectDevice) {
    if (pool_->device_count() < 2) {
        GTEST_SKIP() << "Requires multiple GPUs";
    }

    // Allocate from device 1
    void* ptr = pool_->allocate(2048, 1);
    auto ownership = pool_->get_ownership(ptr);
    EXPECT_EQ(ownership.owning_device, 1);

    // Deallocate - should route to device 1's pool
    EXPECT_NO_THROW(pool_->deallocate(ptr));

    // Verify pointer is no longer tracked
    EXPECT_FALSE(pool_->owns_pointer(ptr));
}

TEST_F(DistributedMemoryPoolTest, UnknownPointerDeallocateThrows) {
    void* unknown_ptr = malloc(1024);  // Not allocated from our pool

    EXPECT_THROW(pool_->deallocate(unknown_ptr), std::invalid_argument);

    free(unknown_ptr);
}

TEST_F(DistributedMemoryPoolTest, DoubleDeallocateThrows) {
    void* ptr = pool_->allocate(1024, 0);
    pool_->deallocate(ptr);

    EXPECT_THROW(pool_->deallocate(ptr), std::invalid_argument);
}

// ============================================================================
// Aggregate Statistics
// ============================================================================

TEST_F(DistributedMemoryPoolTest, AggregateStatistics) {
    // Allocate on multiple devices
    std::vector<void*> ptrs;
    for (int i = 0; i < pool_->device_count(); ++i) {
        ptrs.push_back(pool_->allocate(1024, i));
    }

    auto all_metrics = pool_->get_all_metrics();

    size_t total_allocated = 0;
    for (const auto& metrics : all_metrics) {
        total_allocated += metrics.allocated_bytes;
    }

    // Should have allocated at least one block per device
    EXPECT_GE(total_allocated, pool_->device_count() * 1024);

    // Cleanup
    for (void* ptr : ptrs) {
        pool_->deallocate(ptr);
    }
}

TEST_F(DistributedMemoryPoolTest, ClearAllPools) {
    // Allocate some memory
    std::vector<void*> ptrs;
    for (int i = 0; i < std::min(2, pool_->device_count()); ++i) {
        ptrs.push_back(pool_->allocate(1024, i));
    }

    // Clear all pools
    pool_->clear();

    // All pointers should be invalid
    for (void* ptr : ptrs) {
        EXPECT_FALSE(pool_->owns_pointer(ptr));
    }
}

// ============================================================================
// Single-GPU Fallback Tests (PITFALL-6, PITFALL-9)
// ============================================================================

TEST_F(DistributedMemoryPoolTest, SingleGpuFallback) {
    if (pool_->device_count() > 1) {
        GTEST_SKIP() << "This test is for single-GPU fallback";
    }

    // Should still work on single GPU
    void* ptr = pool_->allocate(1024, 0);
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(pool_->owns_pointer(ptr));

    auto ownership = pool_->get_ownership(ptr);
    EXPECT_EQ(ownership.owning_device, 0);

    pool_->deallocate(ptr);
}

TEST_F(DistributedMemoryPoolTest, IsSingleGpuFlag) {
    if (pool_->device_count() == 1) {
        EXPECT_TRUE(pool_->is_single_gpu());
    } else {
        EXPECT_FALSE(pool_->is_single_gpu());
    }
}

// ============================================================================
// Pool Metrics Tests
// ============================================================================

TEST_F(DistributedMemoryPoolTest, DeviceMetricsIncludeLocalPoolMetrics) {
    void* ptr = pool_->allocate(4096, 0);

    auto metrics = pool_->get_device_metrics(0);
    EXPECT_GT(metrics.local_metrics.hits, 0);  // Should have pool activity

    pool_->deallocate(ptr);
}

TEST_F(DistributedMemoryPoolTest, AllocationCountTracking) {
    int initial_count = pool_->get_device_metrics(0).num_allocations;

    void* ptr1 = pool_->allocate(1024, 0);
    void* ptr2 = pool_->allocate(2048, 0);

    int after_count = pool_->get_device_metrics(0).num_allocations;
    EXPECT_EQ(after_count, initial_count + 2);

    pool_->deallocate(ptr1);
    pool_->deallocate(ptr2);
}

// ============================================================================
// Config Tests
// ============================================================================

TEST_F(DistributedMemoryPoolTest, CustomConfig) {
    DistributedMemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks_per_device = 8;
    config.enable_auto_allocation = false;

    DistributedMemoryPool custom_pool(config);

    // Should have same device count
    EXPECT_EQ(custom_pool.device_count(), pool_->device_count());

    // Auto-allocation should be disabled, so it should fall back to device 0
    void* ptr = custom_pool.allocate_auto(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(custom_pool.owns_pointer(ptr));

    auto ownership = custom_pool.get_ownership(ptr);
    EXPECT_EQ(ownership.owning_device, 0);  // Fallback to device 0

    custom_pool.deallocate(ptr);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(DistributedMemoryPoolTest, MoveConstructor) {
    void* ptr = pool_->allocate(1024, 0);
    EXPECT_TRUE(pool_->owns_pointer(ptr));

    // Store original device count
    int original_device_count = pool_->device_count();

    // Move construct
    DistributedMemoryPool moved_pool(std::move(*pool_));

    // The moved pool should own the pointer and have same device count
    EXPECT_TRUE(moved_pool.owns_pointer(ptr));
    EXPECT_EQ(moved_pool.device_count(), original_device_count);

    // Original pool should be in valid but unspecified state
    // (device_count() will return 0 as pools_ was moved from)

    moved_pool.deallocate(ptr);
}

TEST_F(DistributedMemoryPoolTest, MoveAssignment) {
    void* ptr1 = pool_->allocate(1024, 0);
    EXPECT_TRUE(pool_->owns_pointer(ptr1));

    DistributedMemoryPool other_pool;
    void* ptr2 = other_pool.allocate(2048, 0);
    EXPECT_TRUE(other_pool.owns_pointer(ptr2));

    // Move assign
    *pool_ = std::move(other_pool);

    // pool_ should now own ptr2, not ptr1
    EXPECT_TRUE(pool_->owns_pointer(ptr2));
    EXPECT_FALSE(pool_->owns_pointer(ptr1));

    pool_->deallocate(ptr2);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(DistributedMemoryPoolTest, InvalidDeviceIdThrows) {
    int max_device = pool_->device_count();

    EXPECT_THROW(pool_->allocate(1024, -1), std::invalid_argument);
    EXPECT_THROW(pool_->allocate(1024, max_device), std::invalid_argument);
    EXPECT_THROW(pool_->allocate(1024, max_device + 1), std::invalid_argument);
}

TEST_F(DistributedMemoryPoolTest, InvalidDeviceMetricsThrows) {
    int max_device = pool_->device_count();

    EXPECT_THROW(pool_->get_device_metrics(-1), std::invalid_argument);
    EXPECT_THROW(pool_->get_device_metrics(max_device), std::invalid_argument);
}
