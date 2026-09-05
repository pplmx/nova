#include <cuda/preemption/preemption_handler.h>

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace nova::preemption::test {

namespace {

// Wait for the detached shutdown thread to reach Complete (bounded so a bug
// doesn't hang the suite).
void wait_complete(ShutdownCoordinator& c) {
    for (int i = 0; i < 2000 && !c.is_shutdown_complete(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace

// ShutdownCoordinator and PreemptionManager are process singletons. The
// coordinator's shutdown_in_progress stays true after a request_shutdown
// completes until ::shutdown() resets it, so every test that drives
// request_shutdown MUST finish with wait_complete + coordinator.shutdown() to
// leave the singleton reusable for the next test.

TEST(PreemptionHandlerTest, RemainingTimeoutReportsBudgetWhenIdle) {
    ShutdownCoordinator& c = ShutdownCoordinator::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = false;
    cfg.shutdown_timeout = std::chrono::seconds(5);
    c.initialize(cfg);

    // Idle: the full configured budget, not a stale decremented value.
    EXPECT_EQ(c.get_remaining_timeout(), std::chrono::seconds(5));
    EXPECT_EQ(c.is_shutdown_in_progress(), false);
}

TEST(PreemptionHandlerTest, RemainingTimeoutCountsDownDuringShutdown) {
    ShutdownCoordinator& c = ShutdownCoordinator::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = false;
    cfg.shutdown_timeout = std::chrono::seconds(3);
    c.initialize(cfg);

    c.request_shutdown(SIGTERM);
    const auto initial = c.get_remaining_timeout().count();
    ASSERT_GE(initial, 0);
    ASSERT_LE(initial, 3);

    // Regression (RIL TASK-079, ISS-018): this used to return a frozen value
    // that never counted down. Let ~2s elapse; with a 2s-and-1 margin truncation
    // the remaining budget must strictly shrink.
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    const auto later = c.get_remaining_timeout().count();
    EXPECT_LT(later, initial) << "timeout must actually count down";
    EXPECT_GE(later, 0);

    wait_complete(c);
    c.shutdown();  // leave the singleton reusable
}

TEST(PreemptionHandlerTest, CheckpointCoordinatedRunsCallback) {
    ShutdownCoordinator& c = ShutdownCoordinator::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = true;
    cfg.shutdown_timeout = std::chrono::seconds(5);
    c.initialize(cfg);

    std::atomic<bool> ran{false};
    c.set_checkpoint_callback([&]() { ran.store(true); return true; });

    EXPECT_TRUE(c.checkpoint_coordinated())
        << "callback success must report a successful checkpoint";
    EXPECT_TRUE(ran.load());
}

TEST(PreemptionHandlerTest, CheckpointCoordinatedReportsMissingCallback) {
    ShutdownCoordinator& c = ShutdownCoordinator::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = true;
    cfg.shutdown_timeout = std::chrono::seconds(5);
    c.initialize(cfg);

    // Regression (RIL TASK-079, ISS-018): with checkpointing requested but no
    // save wired, this printed "Saving checkpoint before shutdown" and saved
    // nothing. It must now report the misconfiguration, not pretend.
    c.set_checkpoint_callback(nullptr);
    EXPECT_FALSE(c.checkpoint_coordinated());
}

TEST(PreemptionHandlerTest, CheckpointCoordinatedReportsCallbackFailure) {
    ShutdownCoordinator& c = ShutdownCoordinator::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = true;
    cfg.shutdown_timeout = std::chrono::seconds(5);
    c.initialize(cfg);

    std::atomic<bool> ran{false};
    c.set_checkpoint_callback([&]() { ran.store(true); return false; });

    EXPECT_FALSE(c.checkpoint_coordinated())
        << "a failed durable save must not be reported as success";
    EXPECT_TRUE(ran.load());
}

TEST(PreemptionHandlerTest, CheckpointSkippedWhenDisabled) {
    ShutdownCoordinator& c = ShutdownCoordinator::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = false;
    cfg.shutdown_timeout = std::chrono::seconds(5);
    c.initialize(cfg);

    std::atomic<bool> ran{false};
    c.set_checkpoint_callback([&]() { ran.store(true); return true; });

    EXPECT_TRUE(c.checkpoint_coordinated());  // nothing requested — OK
    EXPECT_FALSE(ran.load()) << "disabled checkpointing must not invoke the save";
}

TEST(PreemptionHandlerTest, ShutdownRefusesResetWhileMidFlight) {
    ShutdownCoordinator& c = ShutdownCoordinator::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = true;
    cfg.shutdown_timeout = std::chrono::seconds(30);
    c.initialize(cfg);

    // A blocking checkpoint callback parks the detached shutdown thread inside
    // checkpoint_coordinated for a deterministic window.
    c.set_checkpoint_callback([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        return true;
    });

    c.request_shutdown(SIGTERM);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Regression (RIL TASK-079, ISS-018): the old unconditional reset raced the
    // finalizing thread. A mid-flight coordinator must refuse the reset.
    c.shutdown();
    EXPECT_TRUE(c.is_shutdown_in_progress())
        << "reset must be refused while a shutdown thread is finalizing";
    EXPECT_NE(c.get_phase(), ShutdownPhase::Idle);

    wait_complete(c);
    c.shutdown();  // completed — the reset is now permitted
    EXPECT_FALSE(c.is_shutdown_in_progress());
}

TEST(PreemptionHandlerTest, ResumeValidatorReportsMissingManifest) {
    // Smoke test for the ResumeValidator pimpl: validate_checkpoint on a
    // nonexistent dir must return a clean invalid result — before the ctor
    // allocated impl_, this method (and every other) dereferenced a null
    // unique_ptr and SEGV'd the process.
    ResumeValidator& v = ResumeValidator::instance();
    v.set_checkpoint_dir("/nonexistent/nova/ckpt");
    auto result = v.validate_checkpoint("/nonexistent/nova/ckpt");
    EXPECT_FALSE(result.is_valid);
    EXPECT_EQ(result.error_message, "Checkpoint manifest not found");
    EXPECT_FALSE(v.recover_state("/nonexistent/nova/ckpt"));
    EXPECT_EQ(v.get_latest_checkpoint_path(), "/nonexistent/nova/ckpt");
}

TEST(PreemptionHandlerTest, ManagerForwardsCheckpointCallback) {
    // End-to-end through the management seam (no real signal needed): the
    // callback registered on PreemptionManager must reach the coordinator's
    // checkpoint_on_shutdown path during a triggered shutdown.
    PreemptionManager& m = PreemptionManager::instance();
    ShutdownConfig cfg;
    cfg.checkpoint_on_shutdown = true;
    cfg.shutdown_timeout = std::chrono::seconds(5);

    m.initialize(cfg);
    EXPECT_TRUE(m.get_status().preemption_handlers_installed);

    std::atomic<bool> ran{false};
    m.set_checkpoint_callback([&]() { ran.store(true); return true; });

    m.on_preemption_signal(SIGTERM);

    wait_complete(ShutdownCoordinator::instance());
    EXPECT_TRUE(ran.load()) << "the manager-registered save must run";
    EXPECT_TRUE(ShutdownCoordinator::instance().is_shutdown_complete());

    m.shutdown();
    ShutdownCoordinator::instance().shutdown();  // leave the singleton reusable
}

}  // namespace nova::preemption::test
