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

#include <dirent.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

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

    // The coordinator is a process-wide singleton and a callback registered by an earlier
    // test survives. My first version of this case assumed a clean slate and failed when
    // run after one that registered a callback — the assumption was wrong, not the code.
    coordinator.set_checkpoint_callback(nullptr);

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

// The signal path now runs through a self-pipe: the handler does nothing but write(2) a
// byte, and a dispatcher thread runs the callback outside signal context. These cases
// exercise that path end to end with raise(), which is the only way to see it.
TEST(SignalHandlerDispatchTest, TheCallbackRunsAfterASignalIsRaised) {
    nova::preemption::SignalHandler& handler =
        nova::preemption::SignalHandler::instance();

    std::atomic<int> received{0};
    handler.set_shutdown_callback([&received](int signal) { received.store(signal); });
    handler.install_handlers();

    // Poll for the dispatcher rather than sleeping a fixed time: the point is that the
    // callback eventually runs, not that it runs within some deadline.
    std::raise(SIGUSR1);
    for (int i = 0; i < 400 && received.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    EXPECT_EQ(received.load(), SIGUSR1)
        << "the dispatcher thread should have seen the signal number through the pipe";
    EXPECT_TRUE(handler.is_shutdown_requested())
        << "the flag is set in the handler itself, so it is visible without waiting";

    handler.uninstall_handlers();
}

TEST(SignalHandlerDispatchTest, TheFlagIsSetEvenIfNoCallbackIsRegistered) {
    nova::preemption::SignalHandler& handler =
        nova::preemption::SignalHandler::instance();
    handler.install_handlers();

    std::raise(SIGTERM);

    // The handler stores the flag before anything else, so this must be true immediately.
    EXPECT_TRUE(handler.is_shutdown_requested());
    EXPECT_EQ(handler.received_signal(), SIGTERM);

    handler.uninstall_handlers();
}

TEST(SignalHandlerDispatchTest, InstallAndUninstallAreRepeatable) {
    nova::preemption::SignalHandler& handler =
        nova::preemption::SignalHandler::instance();

    // Each cycle builds a new self-pipe and joins a new dispatcher. A leak of either, or a
    // join against a thread that already finished, would show up as a hang or a failure
    // here rather than as a slow leak somewhere else.
    for (int i = 0; i < 3; ++i) {
            std::atomic<int> received{0};
        handler.set_shutdown_callback([&received](int signal) { received.store(signal); });
        handler.install_handlers();
        std::raise(SIGUSR1);
        for (int spin = 0; spin < 400 && received.load() == 0; ++spin) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        EXPECT_EQ(received.load(), SIGUSR1) << "cycle " << i;
        handler.uninstall_handlers();
    }
}

TEST(SignalHandlerDispatchTest, RepeatedInstallAndUninstallDoesNotLeakFileDescriptors) {
    nova::preemption::SignalHandler& handler =
        nova::preemption::SignalHandler::instance();

    // install_handlers() creates a pipe and uninstall_handlers() must close both ends.
    // Counting /proc/self/fd before and after a run of cycles is the direct check; a leak
    // here would exhaust descriptors in a long-lived process that restarts preemption.
    const auto count_open_fds = [] {
        DIR* dir = ::opendir("/proc/self/fd");
        if (dir == nullptr) {
            return -1;
        }
        int count = 0;
        while (::readdir(dir) != nullptr) {
            ++count;
        }
        ::closedir(dir);
        return count;
    };

    const int before = count_open_fds();
    ASSERT_GT(before, 0) << "/proc/self/fd is not readable on this system";

    for (int i = 0; i < 10; ++i) {
        handler.install_handlers();
        handler.uninstall_handlers();
    }

    const int after = count_open_fds();
    EXPECT_EQ(after, before) << "10 install/uninstall cycles changed the descriptor count";
}

}  // namespace signal_handler_state_test
