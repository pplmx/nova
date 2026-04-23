#include <gtest/gtest.h>
#include "cuda/performance/device_info.h"
#include "cuda/algo/kernel_launcher.h"

class DeviceInfoTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(DeviceInfoTest, GetDevicePropertiesReturnsValidComputeCapability) {
    auto props = cuda::performance::get_device_properties(0);

    EXPECT_GE(props.compute_capability_major, 6);
    EXPECT_GE(props.compute_capability_minor, 0);
    EXPECT_LE(props.compute_capability_minor, 9);
    EXPECT_GT(strlen(props.name), 0);
    EXPECT_EQ(props.device_id, 0);
}

TEST_F(DeviceInfoTest, OptimalBlockSizeInValidRange) {
    int block_size = cuda::performance::get_optimal_block_size(0);

    EXPECT_GE(block_size, 128);
    EXPECT_LE(block_size, 1024);
    EXPECT_EQ(block_size % 32, 0);
}

TEST_F(DeviceInfoTest, MemoryBandwidthIsPositive) {
    size_t bandwidth = cuda::performance::get_memory_bandwidth_gbps(0);

    EXPECT_GT(bandwidth, 0);
}

TEST_F(DeviceInfoTest, GetCurrentDeviceReturnsValidDevice) {
    auto device = cuda::performance::get_current_device();

    ASSERT_TRUE(device.has_value());
    EXPECT_GE(device.value(), 0);
}

TEST_F(DeviceInfoTest, SetAndGetDeviceRoundTrip) {
    int original_device = 0;
    auto orig = cuda::performance::get_current_device();
    if (orig.has_value()) {
        original_device = orig.value();
    }

    cuda::performance::set_device(0);

    auto current = cuda::performance::get_current_device();
    ASSERT_TRUE(current.has_value());
    EXPECT_EQ(current.value(), 0);
}

TEST_F(DeviceInfoTest, GetDeviceCountReturnsPositive) {
    int count = cuda::performance::get_device_count();

    EXPECT_GE(count, 1);
}

TEST_F(DeviceInfoTest, GlobalMemoryIsPositive) {
    size_t memory = cuda::performance::get_global_memory_bytes(0);

    EXPECT_GT(memory, 0);
}

TEST_F(DeviceInfoTest, MultiprocessorCountIsPositive) {
    int count = cuda::performance::get_multiprocessor_count(0);

    EXPECT_GT(count, 0);
}

class KernelLauncherAutoTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(KernelLauncherAutoTest, AutoBlockSetsCorrectBlockSize) {
    cuda::detail::KernelLauncher launcher;
    launcher.auto_block();

    int optimal = cuda::performance::get_optimal_block_size(0);
    EXPECT_EQ(launcher.get_block().x, static_cast<unsigned int>(optimal));
}

TEST_F(KernelLauncherAutoTest, CalcGridAutoUsesDeviceAwareConfiguration) {
    size_t n = 1000;
    dim3 grid = cuda::detail::calc_grid_auto(n);

    int optimal = cuda::performance::get_optimal_block_size(0);
    EXPECT_EQ(grid.x, static_cast<unsigned int>((n + optimal - 1) / optimal));
}

TEST_F(KernelLauncherAutoTest, CalcGrid1DAutoMatchesCalcGridAuto) {
    size_t n = 5000;

    dim3 grid_auto = cuda::detail::calc_grid_auto(n);
    dim3 grid_1d_auto = cuda::detail::calc_grid_1d_auto(n);

    EXPECT_EQ(grid_auto.x, grid_1d_auto.x);
    EXPECT_EQ(grid_auto.y, grid_1d_auto.y);
    EXPECT_EQ(grid_auto.z, grid_1d_auto.z);
}

TEST_F(KernelLauncherAutoTest, AutoBlockChainsCorrectly) {
    cuda::detail::KernelLauncher launcher;
    launcher.block({128, 1, 1}).auto_block().grid({10, 1, 1});

    int optimal = cuda::performance::get_optimal_block_size(0);
    EXPECT_EQ(launcher.get_block().x, static_cast<unsigned int>(optimal));
    EXPECT_EQ(launcher.get_grid().x, 10u);
}
