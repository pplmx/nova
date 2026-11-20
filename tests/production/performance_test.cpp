#include <gtest/gtest.h>

#include "cuda/production/l2_persistence.h"
#include "cuda/production/priority_stream.h"

namespace {

    void reset() {
        (void)cudaDeviceSynchronize();
    }

}  // namespace

class L2PersistenceTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

TEST_F(L2PersistenceTest, DefaultConstruction) {
    cuda::production::L2PersistenceManager manager;
    EXPECT_FALSE(manager.is_active());
    EXPECT_EQ(manager.persistence_size(), 0u);
}

TEST_F(L2PersistenceTest, SetPersistenceSize) {
    cuda::production::L2PersistenceManager manager;

    auto max_size = manager.max_persistence_size();
    EXPECT_GT(max_size, 0u);

    manager.set_persistence_size(max_size / 2);
    EXPECT_TRUE(manager.is_active());
    EXPECT_GT(manager.persistence_size(), 0u);
}

TEST_F(L2PersistenceTest, RestoreDefaults) {
    cuda::production::L2PersistenceManager manager;

    auto max_size = manager.max_persistence_size();
    manager.set_persistence_size(max_size / 2);

    EXPECT_TRUE(manager.is_active());

    manager.restore_defaults();

    EXPECT_FALSE(manager.is_active());
}

TEST_F(L2PersistenceTest, RAIIBehavior) {
    { cuda::production::ScopedL2Persistence persist(1024 * 1024); }

    cuda::production::L2PersistenceManager after_scope;
    EXPECT_FALSE(after_scope.is_active());
}

TEST_F(L2PersistenceTest, MoveSemantics) {
    cuda::production::L2PersistenceManager manager1;
    manager1.set_persistence_size(1024);

    cuda::production::L2PersistenceManager manager2(std::move(manager1));

    EXPECT_TRUE(manager2.is_active());

    manager2.restore_defaults();
}

class PriorityStreamTest : public ::testing::Test {
protected:
    void SetUp() override { reset(); }
};

TEST_F(PriorityStreamTest, DefaultConstruction) {
    cuda::production::PriorityStreamPool pool;
    EXPECT_EQ(pool.total_available(), 0u);
}

TEST_F(PriorityStreamTest, PoolCreation) {
    cuda::production::PriorityStreamPool pool(4);

    EXPECT_EQ(pool.available_count(cuda::production::StreamPriority::Normal), 4u);
    EXPECT_EQ(pool.available_count(cuda::production::StreamPriority::Low), 4u);
    EXPECT_EQ(pool.available_count(cuda::production::StreamPriority::High), 4u);
}

TEST_F(PriorityStreamTest, AcquireRelease) {
    cuda::production::PriorityStreamPool pool(2);

    auto stream1 = pool.acquire(cuda::production::StreamPriority::Normal);
    EXPECT_TRUE(stream1);
    EXPECT_EQ(stream1.priority(), cuda::production::StreamPriority::Normal);

    EXPECT_EQ(pool.available_count(cuda::production::StreamPriority::Normal), 1u);

    pool.release(std::move(stream1));

    EXPECT_EQ(pool.available_count(cuda::production::StreamPriority::Normal), 2u);
}

TEST_F(PriorityStreamTest, PriorityLevels) {
    cuda::production::PriorityStreamPool pool(2);

    auto low = pool.acquire(cuda::production::StreamPriority::Low);
    auto normal = pool.acquire(cuda::production::StreamPriority::Normal);
    auto high = pool.acquire(cuda::production::StreamPriority::High);

    EXPECT_TRUE(low);
    EXPECT_TRUE(normal);
    EXPECT_TRUE(high);

    EXPECT_EQ(low.priority(), cuda::production::StreamPriority::Low);
    EXPECT_EQ(normal.priority(), cuda::production::StreamPriority::Normal);
    EXPECT_EQ(high.priority(), cuda::production::StreamPriority::High);

    pool.release(std::move(low));
    pool.release(std::move(normal));
    pool.release(std::move(high));
}

TEST_F(PriorityStreamTest, StreamReuse) {
    cuda::production::PriorityStreamPool pool(2);

    auto stream1 = pool.acquire(cuda::production::StreamPriority::Normal);
    auto handle1 = stream1.get();

    pool.release(std::move(stream1));

    auto stream2 = pool.acquire(cuda::production::StreamPriority::Normal);

    EXPECT_EQ(stream2.get(), handle1);
}

TEST_F(PriorityStreamTest, Cleanup) {
    cuda::production::PriorityStreamPool pool(4);

    auto stream = pool.acquire(cuda::production::StreamPriority::Normal);
    // The case is named Cleanup: the acquired stream is the thing cleanup() has to
    // reclaim, so assert it came back before looking at the pool's counters.
    ASSERT_TRUE(static_cast<bool>(stream)) << "acquire must hand out a stream";

    pool.cleanup();

    EXPECT_EQ(pool.total_available(), 0u);
}
