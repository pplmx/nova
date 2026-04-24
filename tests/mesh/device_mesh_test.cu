/**
 * @file device_mesh_test.cu
 * @brief Tests for DeviceMesh, PeerCapabilityMap, PeerCopy components
 *
 * Tests cover:
 * - MGPU-01: Device enumeration and properties query
 * - MGPU-02: Peer access capability between GPU pairs
 * - MGPU-03: Peer access matrix with cached lookup
 * - MGPU-04: Async peer-to-peer copy primitives
 */

#include <gtest/gtest.h>

#include <vector>

#include "cuda/device/error.h"
#include "cuda/memory/buffer.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/mesh/peer_copy.h"
#include "cuda/async/stream_manager.h"

namespace cuda::mesh {

// ============================================================================
// DeviceMesh Test Fixture
// ============================================================================

class DeviceMeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the device mesh before each test
        DeviceMesh::instance().initialize();
    }
};

// ============================================================================
// MGPU-01: Device Enumeration Tests
// ============================================================================

TEST_F(DeviceMeshTest, EnumerateDevices) {
    auto& mesh = DeviceMesh::instance();
    EXPECT_GT(mesh.device_count(), 0) << "Should have at least one GPU";
    EXPECT_LE(mesh.device_count(), 16) << "Reasonable upper bound for GPUs";
}

TEST_F(DeviceMeshTest, GetMeshDevices) {
    auto& mesh = DeviceMesh::instance();
    auto devices = mesh.get_mesh_devices();

    EXPECT_EQ(devices.size(), static_cast<size_t>(mesh.device_count()))
        << "Should return info for all devices";

    for (const auto& info : devices) {
        EXPECT_GE(info.device_id, 0);
        EXPECT_LT(info.device_id, mesh.device_count());
        EXPECT_GT(info.global_memory_bytes, 0)
            << "Device should have some global memory";
        EXPECT_GE(info.compute_capability_major, 0);
        EXPECT_GE(info.compute_capability_minor, 0);
    }
}

TEST_F(DeviceMeshTest, DevicePropertiesValid) {
    auto& mesh = DeviceMesh::instance();
    auto devices = mesh.get_mesh_devices();

    for (const auto& info : devices) {
        // Name should be non-empty for valid devices
        EXPECT_TRUE(info.name[0] != '\0' || info.device_id == 0)
            << "Device name should be populated";

        // Compute capability should be reasonable
        EXPECT_GE(info.compute_capability_major, 6)
            << "Modern CUDA devices should be at least compute 6.0";
    }
}

TEST_F(DeviceMeshTest, SingletonPattern) {
    // Verify singleton behavior
    auto& instance1 = DeviceMesh::instance();
    auto& instance2 = DeviceMesh::instance();

    EXPECT_EQ(&instance1, &instance2)
        << "DeviceMesh::instance() should return the same object";
}

TEST_F(DeviceMeshTest, LazyInitializationIdempotent) {
    auto& mesh = DeviceMesh::instance();

    int count_before = mesh.device_count();

    // Call initialize multiple times - should be idempotent
    mesh.initialize();
    mesh.initialize();

    int count_after = mesh.device_count();
    EXPECT_EQ(count_before, count_after)
        << "Multiple initializations should return same device count";
}

// ============================================================================
// MGPU-02: Peer Access Capability Tests
// ============================================================================

TEST_F(DeviceMeshTest, PeerAccessSelfAccess) {
    auto& mesh = DeviceMesh::instance();

    // Every device can access itself
    for (int i = 0; i < mesh.device_count(); ++i) {
        EXPECT_TRUE(mesh.can_access_peer(i, i))
            << "Device " << i << " should be able to access itself";
    }
}

TEST_F(DeviceMeshTest, PeerAccessBetweenDevices) {
    auto& mesh = DeviceMesh::instance();

    if (mesh.device_count() > 1) {
        // Test at least one pair - should not throw
        EXPECT_NO_THROW({
            bool can_access = mesh.can_access_peer(0, 1);
            // Result is cached but still valid
            (void)can_access;
        });
    }
}

TEST_F(DeviceMeshTest, PeerAccessOutOfBounds) {
    auto& mesh = DeviceMesh::instance();

    // Out of bounds should return false, not crash
    EXPECT_FALSE(mesh.can_access_peer(-1, 0));
    EXPECT_FALSE(mesh.can_access_peer(0, -1));
    EXPECT_FALSE(mesh.can_access_peer(mesh.device_count(), 0));
    EXPECT_FALSE(mesh.can_access_peer(0, mesh.device_count()));
}

// ============================================================================
// MGPU-03: Cached Peer Capability Matrix Tests
// ============================================================================

TEST_F(DeviceMeshTest, PeerCapabilityMapLookup) {
    auto& mesh = DeviceMesh::instance();
    const auto& capabilities = mesh.peer_capabilities();

    // Verify matrix dimensions
    EXPECT_EQ(capabilities.device_count(), mesh.device_count())
        << "Capability matrix should match device count";
}

TEST_F(DeviceMeshTest, PeerCapabilityMapDiagonal) {
    auto& mesh = DeviceMesh::instance();
    const auto& capabilities = mesh.peer_capabilities();

    // Verify diagonal is always true (self-access)
    for (int i = 0; i < capabilities.device_count(); ++i) {
        EXPECT_TRUE(capabilities.can_access(i, i))
            << "Self-access should always be true for device " << i;
    }
}

TEST_F(DeviceMeshTest, PeerCapabilityMapCached) {
    auto& mesh = DeviceMesh::instance();

    // Multiple calls should return same result (cached)
    bool result1 = mesh.can_access_peer(0, 0);
    bool result2 = mesh.can_access_peer(0, 0);

    EXPECT_EQ(result1, result2)
        << "Cached results should be consistent";

    // Verify both are true (self-access)
    EXPECT_TRUE(result1);
    EXPECT_TRUE(result2);
}

// ============================================================================
// MGPU-04: Async Peer Copy Tests
// ============================================================================

class PeerCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        DeviceMesh::instance().initialize();
    }
};

TEST_F(PeerCopyTest, SingleDeviceCopyAsync) {
    PeerCopy copier;

    // Allocate test buffers
    std::vector<float> h_src(1024, 1.0f);
    std::vector<float> h_dst(1024, 0.0f);

    cuda::memory::Buffer<float> d_src(1024);
    cuda::memory::Buffer<float> d_dst(1024);

    // Copy data to device
    d_src.copy_from(h_src.data(), 1024);

    // Single device async copy (no peer involved)
    copier.copy_async(d_dst.data(), d_src.data(), 1024 * sizeof(float),
                      0, 0);

    CUDA_CHECK(cudaStreamSynchronize(0));

    // Copy result back
    d_dst.copy_to(h_dst.data(), 1024);

    // Verify data integrity
    for (size_t i = 0; i < h_dst.size(); ++i) {
        EXPECT_EQ(h_dst[i], 1.0f) << "Mismatch at index " << i;
    }
}

TEST_F(PeerCopyTest, SingleDeviceCopySync) {
    PeerCopy copier;

    std::vector<int> h_src(256, 42);
    std::vector<int> h_dst(256, 0);

    cuda::memory::Buffer<int> d_src(256);
    cuda::memory::Buffer<int> d_dst(256);

    d_src.copy_from(h_src.data(), 256);

    // Synchronous copy
    copier.copy(d_dst.data(), d_src.data(), 256 * sizeof(int), 0, 0);

    d_dst.copy_to(h_dst.data(), 256);

    for (int val : h_dst) {
        EXPECT_EQ(val, 42);
    }
}

TEST_F(PeerCopyTest, PeerAccessAvailableSelfAccess) {
    PeerCopy copier;

    // Self-access should always be available
    EXPECT_TRUE(copier.peer_access_available(0, 0));

    // Test with current device
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    EXPECT_TRUE(copier.peer_access_available(current_device, current_device));
}

TEST_F(PeerCopyTest, PeerAccessAvailableMultiGPU) {
    PeerCopy copier;
    auto& mesh = DeviceMesh::instance();

    if (mesh.device_count() > 1) {
        // Should not throw - just query
        EXPECT_NO_THROW({
            bool available = copier.peer_access_available(0, 1);
            // Result may be true or false depending on hardware
            (void)available;
        });
    }
}

TEST_F(PeerCopyTest, EnablePeerAccessSelfAccess) {
    PeerCopy copier;

    // Enabling self-access should be a no-op, not an error
    EXPECT_NO_THROW(copier.enable_peer_access(0, 0));
}

TEST_F(PeerCopyTest, EnablePeerAccessMultiGPU) {
    PeerCopy copier;
    auto& mesh = DeviceMesh::instance();

    if (mesh.device_count() > 1) {
        // Should not throw if peer access is available
        EXPECT_NO_THROW({
            if (mesh.can_access_peer(0, 1)) {
                copier.enable_peer_access(0, 1);
            }
        });
    }
}

TEST_F(PeerCopyTest, PeerCopyWithStream) {
    PeerCopy copier;
    auto& mesh = DeviceMesh::instance();

    if (mesh.device_count() < 2) {
        GTEST_SKIP() << "Requires multiple GPUs";
    }

    // Allocate buffers
    cuda::memory::Buffer<float> d_src(1024);
    cuda::memory::Buffer<float> d_dst(1024);

    // Fill source with known pattern
    std::vector<float> h_src(1024, 42.0f);
    d_src.copy_from(h_src.data(), 1024);

    // Get stream from StreamManager
    auto stream = cuda::async::global_stream_manager().get_stream();

    if (mesh.can_access_peer(0, 1)) {
        EXPECT_NO_THROW({
            copier.copy_async(d_dst.data(), d_src.data(), 1024 * sizeof(float),
                              1, 0, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
    } else {
        GTEST_SKIP() << "Peer access not available between devices 0 and 1";
    }
}

TEST_F(PeerCopyTest, CopyAsyncPeerAccessNotEnabled) {
    PeerCopy copier;
    auto& mesh = DeviceMesh::instance();

    if (mesh.device_count() > 1 && !mesh.can_access_peer(0, 1)) {
        // Attempting peer copy without enablement should throw
        cuda::memory::Buffer<float> d_src(1024);
        cuda::memory::Buffer<float> d_dst(1024);

        EXPECT_THROW({
            copier.copy_async(d_dst.data(), d_src.data(), 1024 * sizeof(float),
                              1, 0);
        }, cuda::device::CudaException);
    }
}

// ============================================================================
// Single-GPU Fallback Tests (PITFALL-6)
// ============================================================================

TEST_F(DeviceMeshTest, SingleGpuFallback) {
    auto& mesh = DeviceMesh::instance();

    // On single-GPU systems, operations should not crash
    EXPECT_GE(mesh.device_count(), 1)
        << "Should have at least one GPU";

    // Self-access should always work
    EXPECT_TRUE(mesh.can_access_peer(0, 0));

    // Should be able to get mesh devices
    auto devices = mesh.get_mesh_devices();
    EXPECT_GT(devices.size(), 0)
        << "Should have at least one device in mesh";
}

TEST_F(PeerCopyTest, SingleGpuCopy) {
    PeerCopy copier;

    // Should work on single GPU without peer access
    cuda::memory::Buffer<int> d_src(256);
    cuda::memory::Buffer<int> d_dst(256);

    std::vector<int> h_data(256, 123);
    d_src.copy_from(h_data.data(), 256);

    // Self-copy should work without peer access
    copier.copy_async(d_dst.data(), d_src.data(), 256 * sizeof(int), 0, 0);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_result(256);
    d_dst.copy_to(h_result.data(), 256);

    for (int val : h_result) {
        EXPECT_EQ(val, 123);
    }
}

// ============================================================================
// ScopedDevice Tests
// ============================================================================

TEST(ScopedDeviceTest, DeviceSwitchAndRestore) {
    // Save original device
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));

    // Switch to device 0
    {
        ScopedDevice guard(0);
        int current;
        CUDA_CHECK(cudaGetDevice(&current));
        EXPECT_EQ(current, 0);
    }

    // Should be restored
    int after;
    CUDA_CHECK(cudaGetDevice(&after));
    EXPECT_EQ(after, original_device);
}

TEST(ScopedDeviceTest, NestedDeviceSwitch) {
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));

    {
        ScopedDevice outer(0);
        EXPECT_EQ(0, [&]() {
            int d;
            CUDA_CHECK(cudaGetDevice(&d));
            return d;
        }());

        // Nested switch
        {
            ScopedDevice inner(0);  // Same device for safety
            EXPECT_EQ(0, [&]() {
                int d;
                CUDA_CHECK(cudaGetDevice(&d));
                return d;
            }());
        }

        // Should still be on device 0 (outer's scope)
        EXPECT_EQ(0, [&]() {
            int d;
            CUDA_CHECK(cudaGetDevice(&d));
            return d;
        }());
    }

    // Should be restored to original
    int after;
    CUDA_CHECK(cudaGetDevice(&after));
    EXPECT_EQ(after, original_device);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(PeerCopyTest, MultipleCopyOps) {
    PeerCopy copier;

    // Multiple sequential copies should all succeed
    for (int i = 0; i < 10; ++i) {
        cuda::memory::Buffer<float> d_src(256);
        cuda::memory::Buffer<float> d_dst(256);

        std::vector<float> h_data(256, static_cast<float>(i));
        d_src.copy_from(h_data.data(), 256);

        copier.copy_async(d_dst.data(), d_src.data(), 256 * sizeof(float), 0, 0);

        CUDA_CHECK(cudaStreamSynchronize(0));

        std::vector<float> h_result(256);
        d_dst.copy_to(h_result.data(), 256);

        for (float val : h_result) {
            EXPECT_EQ(val, static_cast<float>(i));
        }
    }
}

TEST_F(PeerCopyTest, PeerAccessTracking) {
    PeerCopy copier;

    // First enable should work
    auto& mesh = DeviceMesh::instance();
    if (mesh.device_count() > 1 && mesh.can_access_peer(0, 1)) {
        EXPECT_NO_THROW(copier.enable_peer_access(0, 1));

        // Second enable should be idempotent (no-op)
        EXPECT_NO_THROW(copier.enable_peer_access(0, 1));
    }
}

}  // namespace cuda::mesh
