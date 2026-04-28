#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "cuda/error/timeout_context.hpp"

using namespace nova::error;

class TimeoutPropagationTest : public ::testing::Test {};

TEST_F(TimeoutPropagationTest, ChildInheritsParentDeadline) {
    auto parent_id = timeout_manager::instance().start_operation("parent", std::chrono::milliseconds{100});

    std::this_thread::sleep_for(std::chrono::milliseconds{30});

    timeout_context child(nullptr, std::chrono::milliseconds{0});
    auto remaining = child.remaining();

    EXPECT_LT(remaining.count(), 100);
    EXPECT_GT(remaining.count(), 0);

    timeout_manager::instance().end_operation(parent_id);
}

TEST_F(TimeoutPropagationTest, ExplicitChildTimeoutOverridesParent) {
    auto parent_id = timeout_manager::instance().start_operation("parent", std::chrono::milliseconds{200});

    timeout_context child(nullptr, std::chrono::milliseconds{50});
    auto remaining = child.remaining();

    EXPECT_LT(remaining.count(), 60);

    timeout_manager::instance().end_operation(parent_id);
}

TEST_F(TimeoutPropagationTest, CallbackInvokedOnTimeout) {
    bool callback_invoked = false;

    auto& manager = timeout_manager::instance();
    manager.set_callback([&](operation_id, std::error_code) {
        callback_invoked = true;
    });

    auto id = manager.start_operation("test", std::chrono::milliseconds{5});
    std::this_thread::sleep_for(std::chrono::milliseconds{20});

    EXPECT_TRUE(callback_invoked);
    manager.end_operation(id);
}

TEST_F(TimeoutPropagationTest, ScopedTimeoutManagesLifespan) {
    {
        scoped_timeout guard("test", std::chrono::milliseconds{100});
        EXPECT_GT(guard.context().id(), 0u);
    }

    EXPECT_EQ(timeout_manager::instance().active_count(), 0u);
}
