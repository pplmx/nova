#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <system_error>
#include "cuda/error/timeout_context.hpp"

using namespace nova::error;

class TimeoutPropagationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Isolate each test from the shared singleton: drop operations and
        // callback state left behind by a previous case.
        timeout_manager::instance().reset();
    }
};

TEST_F(TimeoutPropagationTest, ChildInheritsParentDeadline) {
    // The child must be handed the parent context to inherit its deadline; a
    // zero timeout then means "use the parent's remaining budget".
    timeout_context parent(nullptr, std::chrono::milliseconds {100});

    std::this_thread::sleep_for(std::chrono::milliseconds {30});

    timeout_context child(&parent, std::chrono::milliseconds {0});
    auto remaining = child.remaining();

    EXPECT_LT(remaining.count(), 100);
    EXPECT_GT(remaining.count(), 0);
}

// This case was named "ExplicitChildTimeoutOverridesParent" but passed a null parent, so it never
// exercised a parent and a child together and could not have shown which way the two combine. The
// parent it registered was unrelated to the child it built, and the assertion holds whether or not a
// child is constrained by its parent. It is kept for what it does verify -- an explicit timeout is
// honoured when there is no parent -- and renamed for that.
TEST_F(TimeoutPropagationTest, AnExplicitTimeoutIsHonouredWithoutAParent) {
    auto parent_id = timeout_manager::instance().start_operation("parent", std::chrono::milliseconds {200});

    timeout_context child(nullptr, std::chrono::milliseconds {50});
    auto remaining = child.remaining();

    EXPECT_LT(remaining.count(), 60);

    timeout_manager::instance().end_operation(parent_id);
}

// The header says "The effective deadline is min(parent_deadline, start_time + timeout)", and the
// destructor-only parent_ member suggests the parent is meant to constrain the child. The
// implementation consulted the parent only when timeout was zero, so a child asking for a longer
// timeout than its parent had left outlived the parent:
//
//     parent(50ms); child(&parent, 5000ms);
//     measured before this change: child.remaining() = 4999ms, child.is_expired() = 0,
//     and 120ms later the parent had expired while the child still reported 4879ms.
//
// A child that outlives the operation it belongs to is the case hierarchical timeouts exist to
// prevent, so the shorter of the two wins now.
TEST_F(TimeoutPropagationTest, AChildIsNeverGrantedMoreTimeThanItsParentHasLeft) {
    timeout_context parent(nullptr, std::chrono::milliseconds {100});
    timeout_context child(&parent, std::chrono::milliseconds {5000});

    EXPECT_LE(child.remaining().count(), parent.remaining().count()) << "a child asking for 5000ms under a parent with 100ms must not receive more than the parent has";
    EXPECT_LE(child.remaining().count(), 100) << "the documented effective deadline is min(parent, own)";
}

TEST_F(TimeoutPropagationTest, AChildExpiresWhenItsParentExpiresEvenWithALongerOwnTimeout) {
    timeout_context parent(nullptr, std::chrono::milliseconds {30});
    timeout_context child(&parent, std::chrono::milliseconds {5000});

    std::this_thread::sleep_for(std::chrono::milliseconds {120});

    EXPECT_TRUE(parent.is_expired()) << "the parent's 30ms must have elapsed";
    EXPECT_TRUE(child.is_expired()) << "the child must not outlive the parent that bounds it";
}

TEST_F(TimeoutPropagationTest, AChildWithAShorterTimeoutStillUsesItsOwn) {
    // The other half of min(): asking for less than the parent allows must not be extended to the
    // parent's budget.
    timeout_context parent(nullptr, std::chrono::milliseconds {5000});
    timeout_context child(&parent, std::chrono::milliseconds {40});

    EXPECT_LE(child.remaining().count(), 50) << "the child's own, shorter timeout must win";
}

TEST_F(TimeoutPropagationTest, CallbackInvokedOnTimeout) {
    // Delivered on the watchdog thread and read here, so this needs to be atomic: as a plain
    // bool, ThreadSanitizer reports a data race between this case's write in the callback and
    // the assertion below.
    std::atomic<bool> callback_invoked {false};

    auto& manager = timeout_manager::instance();
    manager.set_callback([&](operation_id, std::error_code) { callback_invoked = true; });

    // Callbacks are only delivered by the watchdog loop, so enable it.
    auto config = timeout_config {};
    config.watchdog_interval = std::chrono::milliseconds {10};
    manager.set_config(config);

    const auto id = manager.start_operation("test", std::chrono::milliseconds {5});

    // Wait for the callback rather than for a fixed 30 ms. Delivery depends on the watchdog
    // thread being scheduled, which a loaded machine cannot promise within any fixed budget.
    const auto wait_until = std::chrono::steady_clock::now() + std::chrono::seconds {5};
    while (!callback_invoked.load() && std::chrono::steady_clock::now() < wait_until) {
        std::this_thread::sleep_for(std::chrono::milliseconds {5});
    }

    EXPECT_TRUE(callback_invoked.load()) << "the registered callback should have fired for a 5 ms deadline";
    manager.end_operation(id);
}

TEST_F(TimeoutPropagationTest, ScopedTimeoutManagesLifespan) {
    {
        scoped_timeout guard("test", std::chrono::milliseconds {100});
        EXPECT_GT(guard.context().id(), 0u);
    }

    EXPECT_EQ(timeout_manager::instance().active_count(), 0u);
}

TEST_F(TimeoutPropagationTest, SetDeadlineCallbackInvokesOnTimeout) {
    // The callback runs on the watchdog thread, so these are written there and read here. As
    // plain bool/operation_id that was an unsynchronised read of a concurrently written object.
    std::atomic<bool> callback_invoked {false};
    std::atomic<operation_id> callback_id {0};

    // Callbacks are delivered by the watchdog loop, so it has to be running. The neighbouring
    // CallbackInvokedOnTimeout enables it; this case never did, which is why its callback
    // could not fire and why the conditional assertion below it went unnoticed.
    auto& manager = timeout_manager::instance();
    auto config = timeout_config {};
    config.watchdog_interval = std::chrono::milliseconds {10};
    manager.set_config(config);

    {
        scoped_timeout guard("test", std::chrono::milliseconds {10});
        guard.context().set_deadline_callback([&](operation_id id, std::error_code) {
            callback_invoked = true;
            callback_id = id;
        });

        EXPECT_GT(guard.context().id(), 0u);

        // The wait has to happen while the operation is still registered. It used to sit after
        // the closing brace, so end_operation had already removed the operation from the
        // manager's map and the watchdog had nothing left to expire — the callback could not
        // fire no matter how long the test waited.
        //
        // Wait for the CONDITION, not for a fixed duration. Delivery is done by the watchdog
        // thread, so a fixed 30 ms budget silently assumes the OS schedules that thread inside
        // it; measured, this case failed inside a full-suite run at host load average ~150
        // while passing when run alone. A 5 s ceiling still fails quickly if the callback never
        // fires, and does not depend on how busy the machine is.
        const auto wait_until = std::chrono::steady_clock::now() + std::chrono::seconds {5};
        while (!callback_invoked.load() && std::chrono::steady_clock::now() < wait_until) {
            std::this_thread::sleep_for(std::chrono::milliseconds {5});
        }
    }

    // The case is named for the callback being invoked, so that is what it has to assert. It
    // guarded both checks with `if (callback_invoked)`, so a callback that never fired produced
    // no assertion at all and the case still passed. The neighbour above asserts on its
    // callback directly — the same file, two different standards.
    EXPECT_TRUE(callback_invoked.load()) << "the deadline callback should have fired for a 10 ms deadline";
    EXPECT_GT(callback_id.load(), 0u);
}

TEST_F(TimeoutPropagationTest, TimeoutContextIsExpiredAfterTimeout) {
    {
        scoped_timeout guard("test", std::chrono::milliseconds {5});
        EXPECT_FALSE(guard.context().is_expired());

        std::this_thread::sleep_for(std::chrono::milliseconds {10});
        EXPECT_TRUE(guard.context().is_expired());
    }
}

TEST_F(TimeoutPropagationTest, TimeoutContextRemainingDecreases) {
    scoped_timeout guard("test", std::chrono::milliseconds {50});
    auto initial = guard.context().remaining();

    std::this_thread::sleep_for(std::chrono::milliseconds {20});
    auto later = guard.context().remaining();

    EXPECT_LE(later.count(), initial.count());
}

TEST_F(TimeoutPropagationTest, ContextIdIsValid) {
    scoped_timeout guard("test", std::chrono::milliseconds {100});
    EXPECT_GT(guard.context().id(), 0u);

    auto& manager = timeout_manager::instance();
    EXPECT_TRUE(!manager.is_expired(guard.context().id()) || manager.is_expired(guard.context().id()));
}
