/**
 * @file reduce_kernels_test.cu
 * @brief Contracts for the parallel reduction kernels
 *
 * include/cuda/device/reduce_kernels.h is compiled only by nvcc (it names
 * threadIdx and __syncthreads at namespace scope), so this lives in a .cu file
 * and runs as its own executable — the shared gtest binary is built by the host
 * compiler and cannot host it. It is registered with add_test rather than
 * gtest_discover_tests for the same reason.
 *
 * The kernels had no tests, and both of them were wrong for MAX and MIN:
 *
 *   T val = T{};   // zero
 *
 * Zero is the identity only for SUM. The out-of-range slots of the final block
 * contribute this padding value, so MAX over an all-negative array returned 0
 * (the padding beat every real element) and MIN over an all-positive array did
 * the same. Both kernels now use the operation's own identity.
 */

#include <cuda/device/reduce_kernels.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <climits>
#include <limits>
#include <vector>

namespace cuda::device::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        // Grid size the header documents for these kernels.
        size_t blocks_for(size_t n) {
            return (n + 2 * REDUCE_BLOCK_SIZE - 1) / (2 * REDUCE_BLOCK_SIZE);
        }

        /**
         * @brief Run one reduction kernel over `values` and return the first block result
         *
         * The kernels write one partial per block, so the full result needs a second
         * pass; every case here is small enough to finish in a single block, which keeps
         * the test focused on the kernel's own correctness.
         */
        template <typename T>
        T reduce(const std::vector<T>& values, ReduceOp op, bool optimized) {
            T* device_in = nullptr;
            T* device_out = nullptr;
            const size_t bytes = values.size() * sizeof(T);
            cudaMalloc(&device_in, bytes);
            cudaMemcpy(device_in, values.data(), bytes, cudaMemcpyHostToDevice);

            const size_t grid = blocks_for(values.size());
            cudaMalloc(&device_out, grid * sizeof(T));

            if (optimized) {
                reduce_optimized_kernel<T><<<grid, REDUCE_BLOCK_SIZE>>>(device_in, device_out, values.size(), op);
            } else {
                reduce_basic_kernel<T><<<grid, REDUCE_BLOCK_SIZE>>>(device_in, device_out, values.size(), op);
            }
            cudaDeviceSynchronize();

            T result{};
            cudaMemcpy(&result, device_out, sizeof(T), cudaMemcpyDeviceToHost);
            cudaFree(device_in);
            cudaFree(device_out);
            return result;
        }

    }  // namespace

    // Regression: with the zero padding, the all-negative MAX returned 0.
    TEST(ReduceBasicKernelTest, MaxOverAllNegativeValues) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<float> values = {-5.0f, -3.0f, -9.0f, -1.0f};
        EXPECT_FLOAT_EQ(reduce(values, ReduceOp::MAX, false), -1.0f) << "zero padding must not outrank the real values";
    }

    // Regression: with the zero padding, the all-positive MIN returned 0.
    TEST(ReduceBasicKernelTest, MinOverAllPositiveValues) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<float> values = {5.0f, 3.0f, 9.0f, 1.0f};
        EXPECT_FLOAT_EQ(reduce(values, ReduceOp::MIN, false), 1.0f);
    }

    TEST(ReduceBasicKernelTest, SumIsUnaffected) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_FLOAT_EQ(reduce(values, ReduceOp::SUM, false), 10.0f);
    }

    TEST(ReduceBasicKernelTest, SignedIntegersUseTheirOwnBounds) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<int> negative = {-5, -3, -9, -1};
        EXPECT_EQ(reduce(negative, ReduceOp::MAX, false), -1);

        const std::vector<int> positive = {5, 3, 9, 1};
        EXPECT_EQ(reduce(positive, ReduceOp::MIN, false), 1);
    }

    TEST(ReduceBasicKernelTest, ExtremesAreRepresentable) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        // The identity must be outside the data range without clipping a real value.
        const std::vector<int> values = {INT_MAX, 0, INT_MIN};
        EXPECT_EQ(reduce(values, ReduceOp::MAX, false), INT_MAX);
        EXPECT_EQ(reduce(values, ReduceOp::MIN, false), INT_MIN);
    }

    TEST(ReduceBasicKernelTest, UnsignedTypesUseTheirOwnBounds) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        // An unsigned MAX identity must be 0, not a sign-bit artefact.
        const std::vector<unsigned int> values = {5u, 3u, 9u, 1u};
        EXPECT_EQ(reduce(values, ReduceOp::MAX, false), 9u);
        EXPECT_EQ(reduce(values, ReduceOp::MIN, false), 1u);
    }

    TEST(ReduceBasicKernelTest, SingleElementIsReturnedUnchanged) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<int> values = {-7};
        EXPECT_EQ(reduce(values, ReduceOp::MAX, false), -7);
        EXPECT_EQ(reduce(values, ReduceOp::MIN, false), -7);
        EXPECT_EQ(reduce(values, ReduceOp::SUM, false), -7);
    }

    TEST(ReduceBasicKernelTest, DoublesReduceLikeFloats) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<double> values = {-5.5, -3.5};
        EXPECT_DOUBLE_EQ(reduce(values, ReduceOp::MAX, false), -3.5);
    }

    // ------------------------------------------------ optimized kernel --------
    //
    // Same identity defect, separately instantiated; the shuffle-based path has its
    // own initialisation of the inactive lanes.

    TEST(ReduceOptimizedKernelTest, MaxOverAllNegativeValues) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<float> values = {-5.0f, -3.0f, -9.0f, -1.0f};
        EXPECT_FLOAT_EQ(reduce(values, ReduceOp::MAX, true), -1.0f) << "the inactive-lane value must be MAX's identity, not zero";
    }

    TEST(ReduceOptimizedKernelTest, MinOverAllPositiveValues) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<float> values = {5.0f, 3.0f, 9.0f, 1.0f};
        EXPECT_FLOAT_EQ(reduce(values, ReduceOp::MIN, true), 1.0f);
    }

    TEST(ReduceOptimizedKernelTest, SumMatchesTheBasicKernel) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_FLOAT_EQ(reduce(values, ReduceOp::SUM, true), reduce(values, ReduceOp::SUM, false));
    }

    TEST(ReduceOptimizedKernelTest, AgreesWithTheBasicKernelOnAllOperations) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        const std::vector<int> values = {-4, 8, -2, 16, 0, -32};
        for (const ReduceOp op : {ReduceOp::SUM, ReduceOp::MAX, ReduceOp::MIN}) {
            EXPECT_EQ(reduce(values, op, true), reduce(values, op, false)) << "the two kernels must agree for op " << static_cast<int>(op);
        }
    }

}  // namespace cuda::device::test
