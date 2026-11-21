#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda/memory/distributed_pool.h"
#include "cuda/mesh/device_mesh.h"

#define CUDA_CHECK(call)                            \
    do {                                            \
        cudaError_t err = call;                     \
        if (err != cudaSuccess) {                   \
            throw std::runtime_error("CUDA error"); \
        }                                           \
    } while (0)

using cuda::memory::DistributedMemoryPool;

class DistributedMemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override { pool_ = std::make_unique<DistributedMemoryPool>(); }

    void TearDown() override { pool_.reset(); }

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
    void* ptr1 = pool_->device_count() > 1 ? pool_->allocate(2048, 1) : nullptr;

    // Verify metrics for each device
    auto metrics0 = pool_->get_device_metrics(0);
    EXPECT_GT(metrics0.allocated_bytes, 0);

    if (pool_->device_count() > 1) {
        auto metrics1 = pool_->get_device_metrics(1);
        EXPECT_GT(metrics1.allocated_bytes, 0);
    }

    // Cleanup
    pool_->deallocate(ptr0);
    if (ptr1) {
        pool_->deallocate(ptr1);
    }
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

// Regression: get_best_device() ranked devices by total memory minus this pool's OWN
// allocations, which is not free memory as soon as anything else on the machine holds any.
// Measured with 60 GB of device 0 and 30 GB of device 1 taken by another allocation while
// device 2 was untouched:
//
//     dev 0: real_free=20.1 GB   dev 1: 52.4 GB   dev 2: 84.6 GB
//     get_best_device() = 0      <- the device with the LEAST free memory
//
// Every auto-allocation therefore went to the busiest device, the opposite of the policy the
// method exists to implement. The case above only checked that the id is in range.
//
// NOTE ON DETECTION: on an idle machine every device is equally free and the old ranking
// agrees with the driver by accident, so this case has to CREATE the contention. It takes a
// large block on the device the driver calls emptiest and holds it while asking again; only a
// live query moves its answer.
TEST_F(DistributedMemoryPoolTest, GetBestDevicePicksTheDeviceWithTheMostFreeMemory) {
    const int device_count = pool_->device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "needs at least two devices to have a wrong answer";
    }

    int original = 0;
    ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);

    // Ask the driver what the answer is, independently of the pool.
    const auto free_on = [](int device) -> size_t {
        size_t free_mem = 0, total_mem = 0;
        if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        return free_mem;
    };

    std::vector<size_t> free_per_device(static_cast<size_t>(device_count), 0);
    int emptiest = 0;
    for (int i = 0; i < device_count; ++i) {
        free_per_device[static_cast<size_t>(i)] = free_on(i);
        if (free_per_device[static_cast<size_t>(i)] > free_per_device[static_cast<size_t>(emptiest)]) {
            emptiest = i;
        }
    }
    cudaSetDevice(original);

    const int before = pool_->get_best_device();
    EXPECT_GE(before, 0);
    EXPECT_LT(before, device_count);

    // Make the emptiest device the fullest: hold most of its free memory while asking again.
    ASSERT_EQ(cudaSetDevice(emptiest), cudaSuccess);
    void* hog = nullptr;
    const size_t reserve = (free_per_device[static_cast<size_t>(emptiest)] / 4) * 3;
    const bool reserved = reserve > 0 && cudaMalloc(&hog, reserve) == cudaSuccess;
    cudaSetDevice(original);
    if (!reserved) {
        (void)cudaGetLastError();
        GTEST_SKIP() << "could not reserve memory to create contention";
    }

    // What the driver says now, and what a live query should therefore answer.
    int expected_after = -1;
    size_t expected_free = 0;
    for (int i = 0; i < device_count; ++i) {
        const size_t free_now = free_on(i);
        if (free_now > expected_free) {
            expected_free = free_now;
            expected_after = i;
        }
    }
    cudaSetDevice(original);

    const int chosen = pool_->get_best_device();

    cudaSetDevice(emptiest);
    cudaFree(hog);
    cudaSetDevice(original);

    ASSERT_GE(expected_after, 0);
    EXPECT_EQ(chosen, expected_after) << "get_best_device() returned device " << chosen << " while the driver reported device " << expected_after << " as emptiest, after "
                                      << reserve << " bytes were taken from device " << emptiest;
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

// Regression: the metrics reported free_memory as total - this pool's own allocations. That
// is not free memory; it is wrong by whatever every other tenant on the device holds.
// Measured with an 8 GB allocation from another source on device 0: the pool reported
// 85.1 GB free where the driver said 76.1 GB. The field is advertised as free memory, so it
// now comes from the driver.
TEST_F(DistributedMemoryPoolTest, FreeMemoryMetricMatchesTheDriver) {
    int original = 0;
    cudaGetDevice(&original);

    size_t driver_free = 0, driver_total = 0;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMemGetInfo(&driver_free, &driver_total), cudaSuccess);

    // Take a chunk of memory that the pool knows nothing about, standing in for another
    // tenant. Without it the two definitions agree and the case proves nothing.
    void* outsider = nullptr;
    const size_t outsider_bytes = size_t(2) << 30;  // 2 GB
    const bool took_memory = (cudaMalloc(&outsider, outsider_bytes) == cudaSuccess);
    if (!took_memory) {
        (void)cudaGetLastError();
        cudaSetDevice(original);
        GTEST_SKIP() << "could not reserve memory to distinguish the two definitions";
    }

    ASSERT_EQ(cudaMemGetInfo(&driver_free, &driver_total), cudaSuccess);
    cudaSetDevice(original);

    const auto metrics = pool_->get_device_metrics(0);

    // 64 MB of slack: the pool's own bookkeeping and driver granularity move the number.
    const size_t slack = size_t(64) << 20;
    EXPECT_LE(metrics.free_memory, driver_free + slack) << "reported free_memory " << metrics.free_memory << " exceeds the driver's " << driver_free
                                                        << ": the field is not free memory";
    EXPECT_GE(metrics.free_memory + slack, driver_free) << "reported free_memory " << metrics.free_memory << " is below the driver's " << driver_free;
    EXPECT_EQ(metrics.total_memory, driver_total) << "total_memory must be the device's total, not its free memory";

    cudaSetDevice(0);
    cudaFree(outsider);
    cudaSetDevice(original);
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
