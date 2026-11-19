/**
 * @file auto_block_device_test.cpp
 * @brief KernelLauncher::auto_block must use the device the caller is on
 *
 * include/cuda/algo/kernel_launcher.h documented its parameter as
 *
 *     @param device_id GPU device ID (default: current device)
 *
 * while the declaration was
 *
 *     KernelLauncher& auto_block(int device_id = 0) &;
 *
 * a literal zero. On a machine with more than one kind of GPU, a caller running on
 * device 4 received device 0's optimal block size. The value is harmless on homogeneous
 * hardware — every A100 in this machine reports 512 either way, so the misreport could
 * not be observed as a wrong answer — but the header promised something the code did not
 * do, and on mixed hardware that difference is real.
 *
 * The default is now -1, meaning "resolve the currently selected device", and an explicit
 * non-negative id still selects that device.
 *
 * Stated plainly: these cases pass against the pre-fix source as well. Every A100 in this
 * machine reports an optimal block size of 512, so "device 0's answer" and "the current
 * device's answer" are the same number and no assertion can tell them apart here. The
 * tests pin the contract so a future change has to break them deliberately, and on mixed
 * hardware the difference they describe is real — but they are not a regression test for
 * this defect, and claiming otherwise would be wrong.
 */

#include <cuda/algo/kernel_launcher.h>
#include <cuda/performance/device_info.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace auto_block_device_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

    }  // namespace

    // Regression: the default was a literal device 0.
    TEST(AutoBlockDeviceTest, TheDefaultUsesTheCurrentDevice) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        int original = 0;
        ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);

        // Pick a device that is not 0 when one exists, so "used the current device" and
        // "used device 0" are distinguishable by construction.
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        const int target = count > 1 ? 1 : 0;
        ASSERT_EQ(cudaSetDevice(target), cudaSuccess);

        cuda::detail::KernelLauncher launcher;
        launcher.auto_block();

        const int expected = cuda::performance::get_optimal_block_size(target);
        EXPECT_EQ(static_cast<int>(launcher.get_block().x), expected) << "auto_block() should size for device " << target;

        ASSERT_EQ(cudaSetDevice(original), cudaSuccess);
    }

    TEST(AutoBlockDeviceTest, AnExplicitDeviceIdIsHonoured) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        cuda::detail::KernelLauncher launcher;
        launcher.auto_block(0);

        const int expected = cuda::performance::get_optimal_block_size(0);
        EXPECT_EQ(static_cast<int>(launcher.get_block().x), expected);
    }

    TEST(AutoBlockDeviceTest, TheBlockSizeIsAlwaysPositiveAndBounded) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Whatever the resolution path, the value handed to the launcher has to be a usable
        // block size: positive, and a multiple of the warp size on this hardware.
        cuda::detail::KernelLauncher launcher;
        launcher.auto_block();

        const int block = static_cast<int>(launcher.get_block().x);
        EXPECT_GT(block, 0);
        EXPECT_LE(block, 1024) << "no CUDA device supports a block above 1024 threads";
    }

    TEST(AutoBlockDeviceTest, ChainingStillReturnsTheLauncher) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // auto_block participates in the fluent interface; a change to its body must not
        // break that.
        cuda::detail::KernelLauncher launcher;
        auto& chained = launcher.auto_block().grid({4, 1, 1});

        EXPECT_EQ(&chained, &launcher);
        EXPECT_EQ(launcher.get_grid().x, 4u);
    }

}  // namespace auto_block_device_test
