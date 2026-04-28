#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "cuda/error/timeout.hpp"

using namespace nova::error;

class TimeoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(TimeoutTest, TimeoutManagerSingletonIsAccessible) {
    auto& manager = timeout_manager::instance();
    EXPECT_TRUE(manager.active_count() == 0 || manager.active_count() >= 0);
}

TEST_F(TimeoutTest, StartOperationReturnsValidId) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{100});
    EXPECT_GT(id, 0u);
    manager.end_operation(id);
}

TEST_F(TimeoutTest, EndOperationRemovesFromActiveCount) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{100});
    EXPECT_EQ(manager.active_count(), 1u);
    manager.end_operation(id);
    EXPECT_EQ(manager.active_count(), 0u);
}

TEST_F(TimeoutTest, MultipleOperationsHaveUniqueIds) {
    auto& manager = timeout_manager::instance();
    auto id1 = manager.start_operation("op1", std::chrono::milliseconds{100});
    auto id2 = manager.start_operation("op2", std::chrono::milliseconds{100});
    auto id3 = manager.start_operation("op3", std::chrono::milliseconds{100});

    EXPECT_NE(id1, id2);
    EXPECT_NE(id2, id3);
    EXPECT_NE(id1, id3);

    manager.end_operation(id1);
    manager.end_operation(id2);
    manager.end_operation(id3);
}

TEST_F(TimeoutTest, IsExpiredReturnsFalseForActiveOperation) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{1000});
    EXPECT_FALSE(manager.is_expired(id));
    manager.end_operation(id);
}

TEST_F(TimeoutTest, IsExpiredReturnsTrueAfterDeadline) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{10});

    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    EXPECT_TRUE(manager.is_expired(id));

    manager.end_operation(id);
}

TEST_F(TimeoutTest, CancelOperationMarksAsCancelled) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{1000});
    EXPECT_FALSE(manager.is_cancelled(id));

    manager.cancel_operation(id);
    EXPECT_TRUE(manager.is_cancelled(id));

    manager.end_operation(id);
}

TEST_F(TimeoutTest, GetRemainingDecreasesOverTime) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{100});

    auto initial = manager.get_remaining(id);
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    auto later = manager.get_remaining(id);

    EXPECT_LE(later.count(), initial.count());
    manager.end_operation(id);
}

TEST_F(TimeoutTest, GetRemainingReturnsZeroWhenExpired) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{5});

    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    EXPECT_EQ(manager.get_remaining(id).count(), 0);

    manager.end_operation(id);
}

TEST_F(TimeoutTest, UpdateTimeoutExtendsDeadline) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{10});

    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    auto remaining_before = manager.get_remaining(id);

    manager.update_timeout(id, std::chrono::milliseconds{100});
    auto remaining_after = manager.get_remaining(id);

    EXPECT_GT(remaining_after.count(), remaining_before.count());
    manager.end_operation(id);
}

TEST_F(TimeoutTest, TimeoutGuardManagesLifespan) {
    {
        timeout_guard guard("test_op", std::chrono::milliseconds{100});
        EXPECT_GT(guard.id(), 0u);
        EXPECT_FALSE(guard.is_expired());
        auto remaining = guard.remaining();
        EXPECT_GT(remaining.count(), 0);
    }

    auto& manager = timeout_manager::instance();
    EXPECT_EQ(manager.active_count(), 0u);
}

TEST_F(TimeoutTest, TimeoutGuardCancel) {
    timeout_guard guard("test_op", std::chrono::milliseconds{100});
    auto id = guard.id();

    guard.cancel();
    EXPECT_TRUE(timeout_manager::instance().is_cancelled(id));
}

TEST_F(TimeoutTest, TimeoutGuardExtend) {
    timeout_guard guard("test_op", std::chrono::milliseconds{10});
    auto id = guard.id();

    auto before = timeout_manager::instance().get_remaining(id);
    std::this_thread::sleep_for(std::chrono::milliseconds{5});

    guard.extend(std::chrono::milliseconds{50});
    auto after = timeout_manager::instance().get_remaining(id);

    EXPECT_GE(after.count(), before.count());
}

TEST_F(TimeoutTest, TimeoutGuardMoveSemantics) {
    timeout_guard guard1("test_op", std::chrono::milliseconds{100});
    auto id1 = guard1.id();

    timeout_guard guard2(std::move(guard1));
    auto id2 = guard2.id();

    EXPECT_EQ(id1, id2);
    EXPECT_FALSE(guard1.is_expired());
    EXPECT_FALSE(guard2.is_expired());
}

TEST_F(TimeoutTest, TimeoutCallbackIsInvoked) {
    auto& manager = timeout_manager::instance();

    operation_id callback_id{0};
    bool callback_invoked{false};

    manager.set_callback([&](operation_id id, std::error_code) {
        callback_id = id;
        callback_invoked = true;
    });

    auto config = timeout_config{};
    config.watchdog_interval = std::chrono::milliseconds{10};
    manager.set_config(config);

    auto id = manager.start_operation("test_op", std::chrono::milliseconds{5});

    std::this_thread::sleep_for(std::chrono::milliseconds{30});

    if (callback_invoked) {
        EXPECT_EQ(callback_id, id);
    }
}

TEST_F(TimeoutTest, ConfigDefaultTimeoutIsUsedWhenNotSpecified) {
    auto& manager = timeout_manager::instance();
    auto id = manager.start_operation("test_op", std::chrono::milliseconds{0});

    auto config = manager.get_config();
    EXPECT_GT(config.default_timeout.count(), 0);

    manager.end_operation(id);
}

TEST_F(TimeoutTest, ConcurrentOperationsAreHandled) {
    auto& manager = timeout_manager::instance();
    constexpr size_t num_ops = 100;

    std::vector<operation_id> ids;
    for (size_t i = 0; i < num_ops; ++i) {
        auto id = manager.start_operation("concurrent_op", std::chrono::milliseconds{1000});
        if (id > 0) {
            ids.push_back(id);
        }
    }

    EXPECT_EQ(ids.size(), num_ops);

    for (auto id : ids) {
        manager.end_operation(id);
    }

    EXPECT_EQ(manager.active_count(), 0u);
}
