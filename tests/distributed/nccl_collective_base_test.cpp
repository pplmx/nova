/**
 * @file nccl_collective_base_test.cpp
 * @brief The collective base class must forward to the context, not substitute its own answers
 *
 * NcclCollective is the base of every NCCL collective in this tree (AllGather, AllReduce,
 * Broadcast, Barrier, ReduceScatter) but had no test of its own -- the same search that found
 * every other header referenced from tests turned this one up as the only one that nothing
 * included. Its five accessors are one-line forwards to NcclContext, which is exactly the shape
 * where a copy-paste mistake is invisible: a get_stream() that forwarded to get_comm() would still
 * compile, still return a non-null handle, and only fail much later inside NCCL.
 *
 * current_comm() is the one with a real contract to defend. Its header says to use it instead of
 * hard-coding rank 0's communicator, because collectives are per-rank. A base class that ignored
 * the override and answered with get_comm(0) would work on single-GPU runs and deadlock or corrupt
 * multi-GPU ones.
 */

#include <gtest/gtest.h>

#include "cuda/nccl/nccl_collective.h"
#include "cuda/nccl/nccl_context.h"

namespace cuda::nccl::test {

    namespace {

        /// Exposes the protected accessors so the forwards can be compared against the context.
        class CollectiveProbe : public NcclCollective {
        public:
            explicit CollectiveProbe(NcclContext& ctx)
                : NcclCollective(ctx) {}
        };

    }  // namespace

    class NcclCollectiveBaseTest : public ::testing::Test {
    protected:
        void SetUp() override {
            auto& ctx = NcclContext::instance();
            ctx.initialize();
            ctx_ = &ctx;
        }

        NcclContext* ctx_ {nullptr};
    };

    // has_nccl() must reflect the context, not a compile-time constant. In a build without NCCL the
    // context answers false and the collective must agree; a base class that returned true
    // unconditionally would have callers issuing collectives that cannot work.
    TEST_F(NcclCollectiveBaseTest, HasNcclMatchesTheContext) {
        CollectiveProbe probe(*ctx_);
        EXPECT_EQ(probe.has_nccl(), ctx_->has_nccl()) << "the base class must report the context's availability, not its own";
    }

    TEST_F(NcclCollectiveBaseTest, DeviceCountMatchesTheContext) {
        CollectiveProbe probe(*ctx_);
        EXPECT_EQ(probe.device_count(), ctx_->device_count());
    }

    // get_comm() and get_stream() are per-device lookups. Asking for a device outside the group is
    // an error in the context, and the base class must not swallow or reshape it.
    TEST_F(NcclCollectiveBaseTest, GetCommForADeviceOutsideTheGroupIsRejected) {
        CollectiveProbe probe(*ctx_);
        const int outside = ctx_->device_count() + 7;
        // Cast to void inside the macro: both accessors are [[nodiscard]], and discarding the result
        // to check that it throws is the point here.
        EXPECT_ANY_THROW((void)probe.get_comm(outside)) << "an out-of-range device must not return a communicator";
        EXPECT_ANY_THROW((void)probe.get_stream(outside)) << "an out-of-range device must not return a stream";
    }

    // The accessors are forwards: whatever the context answers for a valid device, the collective
    // must answer identically. This is what catches a forward wired to the wrong accessor.
    TEST_F(NcclCollectiveBaseTest, AccessorsForwardToTheContext) {
        CollectiveProbe probe(*ctx_);
        ASSERT_GT(ctx_->device_count(), 0) << "the context must have at least one device for this comparison";
        EXPECT_EQ(probe.get_comm(0), ctx_->get_comm(0));
        EXPECT_EQ(probe.get_stream(0), ctx_->get_stream(0));
    }

}  // namespace cuda::nccl::test
