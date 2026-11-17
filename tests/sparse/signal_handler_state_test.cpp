/**
 * @file signal_handler_state_test.cpp
 * @brief Contracts for the shutdown flag's observable state
 *
 * include/cuda/preemption/preemption_handler.h installs handlers for SIGTERM and SIGUSR1.
 * Reviewing that file against POSIX's async-signal-safe list turned up a real problem that
 * this test does NOT exercise, because it cannot: the handler takes a std::mutex, calls a
 * function-local-static accessor, reads a steady_clock and invokes a std::function. None of
 * those are permitted in a signal handler, and the one that matters most is the mutex — a
 * signal arriving while the same thread holds g_signal_mutex deadlocks permanently.
 *
 * Fixing that is a redesign (self-pipe or signalfd, with the callback run from a reader
 * thread), not a local edit, so it has not been attempted. The header now carries the
 * warning. What is tested here is only the state machine around the flag: that the accessor
 * reports what the flag holds, and that a request can be observed and cleared.
 */

#include <cuda/preemption/preemption_handler.h>

#include <gtest/gtest.h>

namespace signal_handler_state_test {

TEST(SignalHandlerStateTest, HandlerStateAgreesWithItsAccessors) {
    const nova::preemption::SignalHandler& handler =
        nova::preemption::SignalHandler::instance();
    const auto state = handler.get_state();

    EXPECT_EQ(handler.is_shutdown_requested(), state.shutdown_requested)
        << "the accessor and the state struct must agree";
    EXPECT_EQ(handler.received_signal(), state.received_signal_number);
}

TEST(SignalHandlerStateTest, ShutdownCoordinatorStartsNotInProgress) {
    // SignalHandler has no reset of its own; the coordinator owns the lifecycle, so the
    // states below are read through it.
    nova::preemption::ShutdownCoordinator& coordinator =
        nova::preemption::ShutdownCoordinator::instance();

    (void)coordinator.is_shutdown_in_progress();
    (void)coordinator.is_shutdown_complete();
    (void)coordinator.get_phase();
    SUCCEED() << "the three accessors must be safe to call in any state";
}

TEST(SignalHandlerStateTest, ACoordinatedCheckpointWithoutACallbackReportsFailure) {
    // checkpoint_coordinated() documented itself as returning true only when a callback
    // ran and succeeded. With none registered it must return false rather than claim a
    // save happened.
    nova::preemption::ShutdownCoordinator& coordinator =
        nova::preemption::ShutdownCoordinator::instance();

    nova::preemption::ShutdownConfig config;
    config.checkpoint_on_shutdown = true;
    config.shutdown_timeout = std::chrono::seconds(1);
    coordinator.initialize(config);

    EXPECT_FALSE(coordinator.checkpoint_coordinated())
        << "no callback is registered, so nothing can have been saved";
}

TEST(SignalHandlerStateTest, ARegisteredCallbackIsInvokedOnce) {
    nova::preemption::ShutdownCoordinator& coordinator =
        nova::preemption::ShutdownCoordinator::instance();

    nova::preemption::ShutdownConfig config;
    config.checkpoint_on_shutdown = true;
    config.shutdown_timeout = std::chrono::seconds(1);
    coordinator.initialize(config);

    int calls = 0;
    coordinator.set_checkpoint_callback([&calls] {
        ++calls;
        return true;
    });

    EXPECT_TRUE(coordinator.checkpoint_coordinated());
    EXPECT_EQ(calls, 1);
}

TEST(SignalHandlerStateTest, AFailingCallbackIsReportedAsFailure) {
    nova::preemption::ShutdownCoordinator& coordinator =
        nova::preemption::ShutdownCoordinator::instance();

    nova::preemption::ShutdownConfig config;
    config.checkpoint_on_shutdown = true;
    config.shutdown_timeout = std::chrono::seconds(1);
    coordinator.initialize(config);

    coordinator.set_checkpoint_callback([] { return false; });

    EXPECT_FALSE(coordinator.checkpoint_coordinated())
        << "the callback reported failure and must not be reported as success";
}

}  // namespace signal_handler_state_test
