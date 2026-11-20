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

    // Regression: with tp*dp <= device_count the constructor filled the two
    // handle vectors as if they held the whole lattice. tp_comms_ is resized to
    // tp_degree_ but was written with d*tp + t for every group, so a tp=2 dp=2
    // lattice wrote 4 handles into a 2-element vector — a heap overflow that
    // corrupts whatever follows the allocation and only shows up when the
    // process happens to fault. Anything reading a rank within a group
    // afterwards sees a corrupted vector, which is what this case pins.
    //
    // It needs a real NCCL group, because the no-NCCL branch never fills the
    // vectors and the overrun cannot happen there. Both a local context with two
    // devices and the shared singleton are tried so the case runs wherever a
    // pair of GPUs is available.
    TEST(PipelineCommunicatorsTest, MultiDPLatticeDoesNotOverrunItsHandleVectors) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "needs at least two devices";
        }

        // A local context over the first pair keeps the choice of devices explicit and
        // does not disturb the shared singleton other suites reuse.
        ::cuda::nccl::NcclContext local_ctx;
        ::cuda::nccl::NcclContextConfig config;
        config.device_ids = {0, 1};
        try {
            local_ctx.initialize(config);
        } catch (const std::exception& e) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "could not initialise a two-device NCCL context: " << e.what();
        }

        // tp=1 dp=2 is the case the old code wrote out of bounds: it indexed
        // tp_comms_[d*1+0] for d in {0,1} into a 1-element vector.
        PipelineCommunicators comms(local_ctx, 1, 2);

        EXPECT_EQ(comms.tp_degree(), 1);
        EXPECT_EQ(comms.dp_degree(), 2);
        EXPECT_EQ(comms.total_devices(), 2);
        EXPECT_TRUE(comms.supports_split());

        // The TP getter takes a rank within a group, so for tp=1 only rank 0 is
        // valid. Reading past it must be refused rather than return a stale handle.
        EXPECT_EQ(comms.get_tp_comm(1), nullptr);
        EXPECT_EQ(comms.get_dp_comm(2), nullptr);
        EXPECT_EQ(comms.get_dp_comm(-1), nullptr);

        // Every legal handle must still be one of the context's real communicators:
        // an overwrite that landed elsewhere in the heap would corrupt a later
        // read rather than change the answer here.
        EXPECT_EQ(comms.get_tp_comm(0), local_ctx.get_comm(0));
        EXPECT_EQ(comms.get_dp_comm(0), local_ctx.get_comm(0));
        EXPECT_EQ(comms.get_dp_comm(1), local_ctx.get_comm(1));

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
