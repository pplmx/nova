/**
 * @file syncbn_overflow_test.cpp
 * @brief SyncBatchNorm::backward rejects an element count that would overflow int
 *
 * backward() computes
 *
 *     int n = batch_size * num_features_ * spatial_size;
 *
 * and each kernel then does `if (idx >= n) return;`. If the product overflows, n turns
 * negative, every thread returns immediately, and the launch reports success while computing
 * nothing — the failure mode that does not announce itself. batch_size and spatial_size were
 * already required to be positive in backward(), and num_features_ is checked in initialize(),
 * so the factors are individually sane; the product needed the guard.
 *
 * The guard sits at the top of backward(), ahead of every launch, so it is reachable without
 * valid buffers — and that is the only part of this behaviour that can be tested safely.
 * Calling backward() with null pointers instead faults the CUDA context asynchronously, and
 * that fault is sticky: an earlier version of this file did so and every case after it in the
 * same process failed with "an illegal memory access was encountered" from the constructor.
 * The cases below therefore stop at the guard and never reach a launch.
 */

#include <cuda/neural/sync_batch_norm.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace syncbn_overflow_test {

    namespace {

        bool named_as_overflow(const std::exception& e) {
            return std::string(e.what()).find("overflows int") != std::string::npos;
        }

    }  // namespace

    class SyncBnOverflowTest : public ::testing::Test {
    protected:
        // Anything that touches the driver in a case is cleared here so it cannot leak into the
        // next one. The cases are written not to fault, and this is the belt to that braces.
        void TearDown() override { (void)cudaGetLastError(); }
    };

    TEST_F(SyncBnOverflowTest, ACountPastIntIsNamedAsAnOverflow) {
        // num_features is fixed by the constructor and kept small so the allocations are trivial;
        // the size comes from the call arguments: 2 * 2 * 2^30 = 2^32, past INT_MAX.
        cuda::neural::SyncBatchNorm norm(/*num_features=*/2);

        // Real one-element buffers rather than null pointers. The guard this case exercises
        // runs before any launch, but backward() now rejects null buffers outright, and a
        // null-pointer call is exactly what used to poison the context for every later case
        // in the same process (see the file comment). Small buffers keep the case honest
        // without depending on the fault path at all.
        float placeholder = 0.0f;
        auto call = [&] {
            norm.backward(&placeholder,
                          &placeholder,
                          &placeholder,
                          &placeholder,
                          &placeholder,
                          /*batch_size=*/2,
                          /*spatial_size=*/1073741824);
        };

        bool reported_overflow = false;
        try {
            call();
        } catch (const std::exception& e) {
            reported_overflow = named_as_overflow(e);
        }
        EXPECT_TRUE(reported_overflow) << "the count must be rejected by name before any launch is attempted";
    }

    TEST_F(SyncBnOverflowTest, ACountInsideIntIsNotReportedAsAnOverflow) {
        // 2 * 2 * 4 = 16 elements, which does not overflow. This case only checks that the
        // overflow message is not raised; it says nothing about the launch that follows,
        // and must not reach one -- see the file comment on why a null-buffer call here
        // was the thing that broke every later case in the process.
        cuda::neural::SyncBatchNorm norm(/*num_features=*/2);

        // A 16-element scratch buffer on the device, so a launch that does happen reads and
        // writes real memory instead of address zero.
        constexpr int kElements = 16;
        float* d_scratch = nullptr;
        ASSERT_EQ(cudaMalloc(&d_scratch, kElements * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_scratch, 0, kElements * sizeof(float)), cudaSuccess);

        auto call = [&] {
            norm.backward(d_scratch,
                          d_scratch,
                          d_scratch,
                          d_scratch,
                          d_scratch,
                          /*batch_size=*/2,
                          /*spatial_size=*/4);
        };

        bool reported_overflow = false;
        try {
            call();
        } catch (const std::exception& e) {
            reported_overflow = named_as_overflow(e);
        }
        EXPECT_FALSE(reported_overflow) << "16 elements do not overflow int";
        (void)cudaGetLastError();

        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "the launch must not fault: a sticky illegal access here would break every "
                                                           "later case in this process";
        EXPECT_EQ(cudaFree(d_scratch), cudaSuccess);
    }

}  // namespace syncbn_overflow_test
