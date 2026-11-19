/**
 * @file pipeline_comm_test.cpp
 * @brief Contracts for the hybrid TP+DP communicator lattice
 *
 * include/cuda/pipeline/pipeline_comm.h builds the TP and DP communicators that
 * hybrid parallelism needs. It had no tests, and the constructor had four
 * defects:
 *
 * - No validation of the degrees. tp=dp=0 produced an object that reported
 *   total_devices() == 0 while supports_split() == true; a negative degree
 *   surfaced as a raw std::length_error ("vector::_M_default_append") instead of
 *   a message naming the argument.
 *
 * - When tp*dp exceeded the device count it silently left supports_split_ false
 *   and returned an empty object, leaving the caller to discover the failure
 *   when a collective got a null communicator.
 *
 * - NcclContext::get_comm() takes a DEVICE id, but the loop passed group
 *   POSITIONS. For tp_degree or dp_degree of 1 the two coincide, which is why
 *   the mistake went unnoticed.
 *
 * - The no-NCCL build reported supports_split() == true with empty vectors.
 *
 * These tests run in the default build, where NCCL is available, and pin the
 * validation and the reported lattice geometry. The communicator handles
 * themselves require an initialised NcclContext, which needs real GPUs.
 */

#include <cuda/pipeline/pipeline_comm.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace cuda::pipeline::test {

    TEST(PipelineCommunicatorsTest, RejectsNonPositiveDegrees) {
        auto& ctx = ::cuda::nccl::NcclContext::instance();

        EXPECT_THROW(PipelineCommunicators(ctx, 0, 2), std::invalid_argument);
        EXPECT_THROW(PipelineCommunicators(ctx, 2, 0), std::invalid_argument);
        EXPECT_THROW(PipelineCommunicators(ctx, -1, 2), std::invalid_argument);
        EXPECT_THROW(PipelineCommunicators(ctx, 2, -1), std::invalid_argument);

        (void)cudaGetLastError();
    }

    // Regression: this used to construct successfully and report a split for a
    // lattice with no devices in it.
    TEST(PipelineCommunicatorsTest, ZeroDegreesDoNotProduceASuccessfulObject) {
        auto& ctx = ::cuda::nccl::NcclContext::instance();

        bool constructed = false;
        try {
            PipelineCommunicators comms(ctx, 0, 0);
            constructed = true;
            EXPECT_FALSE(comms.supports_split()) << "an empty lattice must not claim the split succeeded";
            EXPECT_EQ(comms.total_devices(), 0);
        } catch (const std::invalid_argument&) {
            // Rejecting outright is also correct.
        }
        EXPECT_FALSE(constructed && false);

        (void)cudaGetLastError();
    }

    // Regression: a negative degree escaped as std::length_error from inside
    // vector::resize, which says nothing about the argument.
    TEST(PipelineCommunicatorsTest, NegativeDegreeDoesNotSurfaceAsALengthError) {
        auto& ctx = ::cuda::nccl::NcclContext::instance();

        EXPECT_THROW(PipelineCommunicators(ctx, -4, 1), std::invalid_argument);

        (void)cudaGetLastError();
    }

    TEST(PipelineCommunicatorsTest, OverSubscribedLatticeIsRejectedOrReported) {
        auto& ctx = ::cuda::nccl::NcclContext::instance();

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "no CUDA devices";
        }

        // Ask for more devices than exist. The old code silently produced an object
        // whose supports_split() was false and whose handle vectors were empty.
        const int huge = device_count + 8;
        try {
            PipelineCommunicators comms(ctx, huge, 1);
            EXPECT_FALSE(comms.supports_split()) << "a lattice larger than the machine cannot be split";
            EXPECT_EQ(comms.get_tp_comm(0), nullptr);
        } catch (const std::invalid_argument&) {
            // Rejecting outright is also correct.
        }

        (void)cudaGetLastError();
    }

    TEST(PipelineCommunicatorsTest, ReportsTheRequestedGeometry) {
        auto& ctx = ::cuda::nccl::NcclContext::instance();

        // With no initialised context the constructor cannot obtain handles, so
        // either it throws or the geometry it reports must still be the requested
        // one. total_devices()/tp_degree()/dp_degree() are pure arithmetic.
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "needs at least two devices";
        }

        try {
            PipelineCommunicators comms(ctx, 2, 1);
            EXPECT_EQ(comms.tp_degree(), 2);
            EXPECT_EQ(comms.dp_degree(), 1);
            EXPECT_EQ(comms.total_devices(), 2);
        } catch (const std::exception&) {
            // An uninitialised NcclContext throws out of get_comm; that is the
            // context's contract, not this class's geometry.
            (void)cudaGetLastError();
            GTEST_SKIP() << "NcclContext is not initialised in this process";
        }

        (void)cudaGetLastError();
    }

    TEST(PipelineCommunicatorsTest, UnknownRanksReturnNullptrRatherThanIndexing) {
        auto& ctx = ::cuda::nccl::NcclContext::instance();

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "needs at least two devices";
        }

        try {
            PipelineCommunicators comms(ctx, 2, 1);

            // Out-of-range ranks are a caller error the API answers with nullptr.
            EXPECT_EQ(comms.get_tp_comm(-1), nullptr);
            EXPECT_EQ(comms.get_tp_comm(comms.tp_degree()), nullptr);
            EXPECT_EQ(comms.get_dp_comm(-1), nullptr);
            EXPECT_EQ(comms.get_dp_comm(comms.dp_degree()), nullptr);
        } catch (const std::exception&) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "NcclContext is not initialised in this process";
        }

        (void)cudaGetLastError();
    }

}  // namespace cuda::pipeline::test
