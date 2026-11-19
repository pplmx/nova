/**
 * @file device_pointer_contract_test.cpp
 * @brief The functions that take raw device pointers must be documented and work that way
 *
 * cuda::algo::reduce_sum handed the caller's pointer straight to a CUDA kernel with no
 * staging, and its header carried no comment saying so. A fuzz target in this tree called it
 * with std::vector::data() and faulted — found when that target was rewritten to call the
 * library instead of its own loop.
 *
 * These cases pin the contract from the working side: staging host data into a
 * cuda::memory::Buffer and passing buffer.data() produces the right answer. The failing side
 * — passing a host pointer — would fault the process rather than fail an assertion, so it is
 * described here instead of executed.
 */

#include <cuda/algo/reduce.h>
#include <cuda/memory/buffer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <numeric>
#include <vector>

namespace device_pointer_contract_test {

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

    TEST(DevicePointerContractTest, ReduceSumWorksOnDeviceData) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> host{1.0F, 2.0F, 3.0F, 4.0F};

        // The documented way: stage onto the device, then pass the device pointer.
        cuda::memory::Buffer<float> device(host.size());
        device.copy_from(host.data(), host.size());

        const float sum = cuda::algo::reduce_sum(device.data(), host.size());
        EXPECT_FLOAT_EQ(sum, 10.0F) << "the staged path is the supported one";
    }

    TEST(DevicePointerContractTest, ReduceMaxAndMinAgreeOnTheSameDeviceData) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<float> host{-5.0F, 3.0F, 9.0F, 0.0F};
        cuda::memory::Buffer<float> device(host.size());
        device.copy_from(host.data(), host.size());

        EXPECT_FLOAT_EQ(cuda::algo::reduce_max(device.data(), host.size()), 9.0F);
        EXPECT_FLOAT_EQ(cuda::algo::reduce_min(device.data(), host.size()), -5.0F);
    }

    TEST(DevicePointerContractTest, AnEmptyDeviceBufferReducesToTheIdentity) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Size zero is the boundary the kernels have to handle without launching a bad grid.
        cuda::memory::Buffer<float> device(0);
        EXPECT_NO_THROW({
            const float sum = cuda::algo::reduce_sum(device.data(), 0);
            EXPECT_EQ(sum, 0.0F) << "an empty sum is the additive identity";
        });
    }

}  // namespace device_pointer_contract_test
