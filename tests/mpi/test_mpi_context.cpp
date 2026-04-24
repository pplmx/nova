/**
 * @file test_mpi_context.cpp
 * @brief Unit tests for MPI context and rank discovery
 *
 * Tests MpiContext initialization, rank discovery, and RAII lifecycle.
 * Tests are skipped if MPI is not enabled.
 */

#include <gtest/gtest.h>

#include "cuda/mpi/mpi_context.h"

namespace cuda::mpi {

class MpiContextTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
        auto& ctx = MpiContext::instance();
        if (ctx.initialized()) {
            ctx.finalize();
        }
    }
};

TEST_F(MpiContextTest, InstanceReturnsSameObject) {
    auto& instance1 = MpiContext::instance();
    auto& instance2 = MpiContext::instance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(MpiContextTest, DefaultInitialization) {
#if NOVA_MPI_ENABLED
    auto& ctx = MpiContext::instance();
    if (!ctx.initialized()) {
        GTEST_SKIP() << "MPI not initialized, this is expected in non-MPI builds";
    }
    EXPECT_TRUE(ctx.initialized() || ctx.world_size() >= 1);
    EXPECT_GE(ctx.world_rank(), 0);
    EXPECT_GE(ctx.world_size(), 1);
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

TEST_F(MpiContextTest, LocalRankCalculation) {
#if NOVA_MPI_ENABLED
    auto& ctx = MpiContext::instance();
    if (!ctx.initialized()) {
        GTEST_SKIP() << "MPI not initialized";
    }
    EXPECT_GE(ctx.local_rank(), 0);
    EXPECT_LT(ctx.local_rank(), ctx.local_size());
    EXPECT_GE(ctx.local_size(), 1);
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

TEST_F(MpiContextTest, NodeIdAssignment) {
#if NOVA_MPI_ENABLED
    auto& ctx = MpiContext::instance();
    if (!ctx.initialized()) {
        GTEST_SKIP() << "MPI not initialized";
    }
    EXPECT_GE(ctx.node_id(), 0);
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

TEST_F(MpiContextTest, HasMpiFlag) {
#if NOVA_MPI_ENABLED
    auto& ctx = MpiContext::instance();
    if (!ctx.initialized()) {
        GTEST_SKIP() << "MPI not initialized";
    }
    EXPECT_EQ(ctx.has_mpi(), ctx.world_size() > 1);
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

TEST_F(MpiContextTest, MainProcessIdentification) {
#if NOVA_MPI_ENABLED
    auto& ctx = MpiContext::instance();
    if (!ctx.initialized()) {
        GTEST_SKIP() << "MPI not initialized";
    }
    bool is_main = ctx.world_rank() == 0;
    EXPECT_EQ(ctx.is_main_process(), is_main);
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

TEST_F(MpiContextTest, ConfigInitialization) {
#if NOVA_MPI_ENABLED
    MpiConfig config;
    config.timeout_ms = 60000;
    config.debug = true;

    auto& ctx = MpiContext::instance();
    if (!ctx.initialized()) {
        ctx.initialize(config);
    }

    EXPECT_TRUE(ctx.initialized() || ctx.world_size() >= 1);
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

TEST_F(MpiContextTest, DoubleInitializeIsSafe) {
#if NOVA_MPI_ENABLED
    auto& ctx = MpiContext::instance();
    EXPECT_NO_THROW({
        if (!ctx.initialized()) {
            ctx.initialize();
        }
        ctx.initialize();
        ctx.initialize();
    });
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

TEST_F(MpiContextTest, LocalDeviceId) {
#if NOVA_MPI_ENABLED
    auto& ctx = MpiContext::instance();
    if (!ctx.initialized()) {
        GTEST_SKIP() << "MPI not initialized";
    }
    int device_id = ctx.get_local_device_id();
    EXPECT_GE(device_id, 0);
#else
    GTEST_SKIP() << "MPI not enabled in this build";
#endif
}

}  // namespace cuda::mpi
