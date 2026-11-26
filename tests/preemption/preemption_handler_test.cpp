#include <cuda/preemption/preemption_handler.h>

#include "test_temp_paths.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
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

        // Regression: this used to return a frozen value
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

        std::atomic<bool> ran {false};
        c.set_checkpoint_callback([&]() {
            ran.store(true);
            return true;
        });

        EXPECT_TRUE(c.checkpoint_coordinated()) << "callback success must report a successful checkpoint";
        EXPECT_TRUE(ran.load());
    }

    // set_stage_callback had no test at all, and the callback it stores was read by the shutdown
    // worker without the mutex that the setter takes:
    //
    //     void ShutdownCoordinator::set_stage_callback(ShutdownStageCallback callback) {
    //         std::lock_guard lock(impl_->callback_mutex);
    //         impl_->stage_callback = std::move(callback);
    //     }
    //     // worker thread, three sites, no lock:
    //     if (impl_->stage_callback) impl_->stage_callback(ShutdownPhase::Signaling);
    //
    // checkpoint_callback, thirty lines above in the same class, already took its copy under that
    // mutex; stage_callback did not. This case drives the three phases while another thread keeps
    // re-registering the callback, which is the interleaving that makes it a data race rather than
    // a theoretical one. TSan is what reports it -- in an ordinary build this case only pins that
    // the phases still run and the callback still fires.
    TEST(PreemptionHandlerTest, StageCallbackIsSafeToReplaceWhileShutdownRuns) {
        ShutdownCoordinator& c = ShutdownCoordinator::instance();
        ShutdownConfig cfg;
        cfg.checkpoint_on_shutdown = false;
        cfg.shutdown_timeout = std::chrono::seconds(5);
        c.initialize(cfg);

        std::atomic<int> fired {0};
        c.set_stage_callback([&fired](ShutdownPhase) { fired.fetch_add(1); });

        std::atomic<bool> stop {false};
        std::thread registrar([&c, &stop] {
            while (!stop.load(std::memory_order_relaxed)) {
                c.set_stage_callback([](ShutdownPhase) {});
                c.set_stage_callback(nullptr);
            }
        });

        for (int i = 0; i < 20; ++i) {
            c.begin_graceful_shutdown();
            c.finalize_shutdown();
        }

        stop.store(true, std::memory_order_relaxed);
        registrar.join();
        c.set_stage_callback(nullptr);

        // finalize_shutdown() leaves the singleton marked complete, which a later case reads. The
        // file's convention for request_shutdown() is wait_complete + shutdown(); these phases set
        // the same flag, so the same reset is owed.
        c.shutdown();

        // Reaching here means the phases ran to completion against a callback being replaced.
        SUCCEED() << "three shutdown phases survived 20 rounds against a re-registering thread";
    }

    TEST(PreemptionHandlerTest, CheckpointCoordinatedReportsMissingCallback) {
        ShutdownCoordinator& c = ShutdownCoordinator::instance();
        ShutdownConfig cfg;
        cfg.checkpoint_on_shutdown = true;
        cfg.shutdown_timeout = std::chrono::seconds(5);
        c.initialize(cfg);

        // Regression: with checkpointing requested but no
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

        std::atomic<bool> ran {false};
        c.set_checkpoint_callback([&]() {
            ran.store(true);
            return false;
        });

        EXPECT_FALSE(c.checkpoint_coordinated()) << "a failed durable save must not be reported as success";
        EXPECT_TRUE(ran.load());
    }

    TEST(PreemptionHandlerTest, CheckpointSkippedWhenDisabled) {
        ShutdownCoordinator& c = ShutdownCoordinator::instance();
        ShutdownConfig cfg;
        cfg.checkpoint_on_shutdown = false;
        cfg.shutdown_timeout = std::chrono::seconds(5);
        c.initialize(cfg);

        std::atomic<bool> ran {false};
        c.set_checkpoint_callback([&]() {
            ran.store(true);
            return true;
        });

        EXPECT_TRUE(c.checkpoint_coordinated());  // nothing requested — OK
        EXPECT_FALSE(ran.load()) << "disabled checkpointing must not invoke the save";
    }

    TEST(PreemptionHandlerTest, ShutdownRefusesResetWhileMidFlight) {
        ShutdownCoordinator& c = ShutdownCoordinator::instance();
        ShutdownConfig cfg;
        cfg.checkpoint_on_shutdown = true;
        cfg.shutdown_timeout = std::chrono::seconds(30);
        c.initialize(cfg);

        // A blocking checkpoint callback parks the shutdown worker inside
        // checkpoint_coordinated for a deterministic window. (The worker is joined, not detached,
        // since that change: a detached worker let ~Impl destroy the condvar underneath it.)
        c.set_checkpoint_callback([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            return true;
        });

        c.request_shutdown(SIGTERM);

        // Wait for the worker to actually leave Idle before asserting anything about its phase.
        // A fixed 50 ms sleep assumed it got scheduled that fast; under a load average in the
        // hundreds it does not, and this case failed in a full-suite run while passing alone.
        //
        // The condition is get_phase() != Idle, NOT is_shutdown_in_progress(): request_shutdown
        // publishes shutdown_in_progress before it even starts the worker thread (deliberately, so
        // that a reader seeing it also sees the deadline fields), whereas phase only becomes
        // Signaling once the thread runs. Waiting on in_progress therefore returns while phase is
        // still Idle, which is exactly the assertion below -- I made that mistake first and the
        // case then failed on every run.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (c.get_phase() == ShutdownPhase::Idle && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_NE(c.get_phase(), ShutdownPhase::Idle) << "the shutdown worker never left Idle within the allowance";

        // Regression: the old unconditional reset raced the
        // finalizing thread. A mid-flight coordinator must refuse the reset.
        c.shutdown();
        EXPECT_TRUE(c.is_shutdown_in_progress()) << "reset must be refused while a shutdown thread is finalizing";
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

    // validate_checkpoint() called std::stoi on the manifest's step line without checking
    // that the line exists, and without bounding the value. All three public entry points --
    // validate_checkpoint, recover_state, attempt_recovery -- threw out of functions whose
    // contract is to REPORT an unusable checkpoint. Measured on the unfixed code:
    //
    //     empty manifest   -> std::invalid_argument: stoi
    //     one line only    -> std::invalid_argument: stoi
    //     non-numeric step -> std::invalid_argument: stoi
    //     huge step        -> std::out_of_range: stoi
    //     negative step    -> valid = true, step = -5
    TEST(PreemptionHandlerTest, MalformedManifestIsReportedNotThrown) {
        struct Case {
            const char* label;
            const char* manifest;
        };
        const Case cases[] = {
            {"empty", ""},
            {"version only", "1\n"},
            {"non-numeric step", "1\nnot-a-number\n"},
            {"step past an int", "1\n99999999999999999999\n"},
            {"negative step", "1\n-5\n"},
            {"step then garbage", "1\n42abc\n"},
        };

        ResumeValidator& v = ResumeValidator::instance();
        const std::string root = nova::test::temp_path("nova-preemption-manifest", "");
        std::filesystem::remove_all(root);
        int index = 0;
        for (const auto& c : cases) {
            const std::string dir = root + "-case" + std::to_string(index++);
            std::filesystem::remove_all(dir);
            std::filesystem::create_directories(dir);
            std::ofstream(dir + "/manifest.txt") << c.manifest;
            std::ofstream(dir + "/model.bin", std::ios::binary) << "weights";
            std::ofstream(dir + "/optimizer.bin", std::ios::binary) << "state";

            // Neither throwing nor reporting the checkpoint as usable.
            nova::preemption::ResumeValidator::ValidationResult result;
            EXPECT_NO_THROW(result = v.validate_checkpoint(dir)) << "validate_checkpoint threw on a " << c.label << " manifest";
            EXPECT_FALSE(result.is_valid) << c.label << " was accepted as a resumable checkpoint";
            EXPECT_FALSE(result.error_message.empty()) << c.label << " was rejected without saying why";
            EXPECT_EQ(result.checkpoint_step, -1) << c.label << " left a step a caller could resume from";

            // The two wrappers must inherit the rejection rather than throwing.
            EXPECT_NO_THROW({ EXPECT_FALSE(v.recover_state(dir)); }) << "recover_state threw on a " << c.label << " manifest";
            nova::preemption::ResumeValidator::RecoveryResult recovery;
            EXPECT_NO_THROW(recovery = v.attempt_recovery(dir)) << "attempt_recovery threw on a " << c.label << " manifest";
            EXPECT_FALSE(recovery.success) << c.label << " was recovered from";
        }
    }

    // A well-formed manifest must still be accepted, or the rejection above is just a wall.
    TEST(PreemptionHandlerTest, WellFormedManifestIsAccepted) {
        ResumeValidator& v = ResumeValidator::instance();
        const std::string dir = nova::test::temp_path("nova-preemption-valid", "");
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        // The layout CheckpointManager::save() writes: version, step, timestamp, counts.
        std::ofstream(dir + "/manifest.txt") << "1\n42\n1700000000000000000\n3\n2";
        std::ofstream(dir + "/model.bin", std::ios::binary) << "weights";
        std::ofstream(dir + "/optimizer.bin", std::ios::binary) << "state";

        const auto result = v.validate_checkpoint(dir);
        EXPECT_TRUE(result.is_valid);
        EXPECT_EQ(result.checkpoint_step, 42);
        EXPECT_TRUE(result.has_model_state);
        EXPECT_TRUE(result.has_optimizer_state);
        EXPECT_TRUE(v.recover_state(dir));

        const auto recovery = v.attempt_recovery(dir);
        EXPECT_TRUE(recovery.success);
        EXPECT_EQ(recovery.recovered_step, 42);
    }

    // Presence of the file is not the same as presence of state: a zero-length model.bin is
    // what a failed or truncated write leaves behind, and ifstream::good() reports it as fine.
    // Measured before this check: two 0-byte files were accepted as a valid checkpoint.
    TEST(PreemptionHandlerTest, EmptyStateFilesAreNotResumableState) {
        ResumeValidator& v = ResumeValidator::instance();
        const std::string dir = nova::test::temp_path("nova-preemption-empty-state", "");
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::ofstream(dir + "/manifest.txt") << "1\n42\n";
        std::ofstream(dir + "/model.bin", std::ios::binary);      // 0 bytes
        std::ofstream(dir + "/optimizer.bin", std::ios::binary);  // 0 bytes

        const auto result = v.validate_checkpoint(dir);
        EXPECT_FALSE(result.is_valid) << "two empty state files are not a checkpoint";
        EXPECT_FALSE(result.has_model_state);
        EXPECT_FALSE(result.has_optimizer_state);
        EXPECT_FALSE(v.recover_state(dir));

        // One non-empty side is still not a complete checkpoint.
        std::ofstream(dir + "/model.bin", std::ios::binary) << "weights";
        const auto partial = v.validate_checkpoint(dir);
        EXPECT_FALSE(partial.is_valid);
        EXPECT_TRUE(partial.has_model_state);
        EXPECT_FALSE(partial.has_optimizer_state);
    }

    // get_status() dereferenced signal_handler, which only initialize() assigns. Measured on a
    // freshly constructed manager -- exactly when a caller asks "what is my state?" before
    // initializing anything -- it crashed with SIGSEGV.
    TEST(PreemptionHandlerTest, GetStatusBeforeInitializeReportsUninitialized) {
        // Called before any initialize() in this process. The singleton is shared with other
        // cases, so this is only a meaningful regression probe when it has not been set up;
        // either way it must not dereference null.
        PreemptionManager& m = PreemptionManager::instance();
        nova::preemption::PreemptionManager::Status status;
        EXPECT_NO_THROW(status = m.get_status());
        EXPECT_FALSE(status.shutdown_in_progress);
        EXPECT_FALSE(status.shutdown_complete);
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

        std::atomic<bool> ran {false};
        m.set_checkpoint_callback([&]() {
            ran.store(true);
            return true;
        });

        m.on_preemption_signal(SIGTERM);

        wait_complete(ShutdownCoordinator::instance());
        EXPECT_TRUE(ran.load()) << "the manager-registered save must run";
        EXPECT_TRUE(ShutdownCoordinator::instance().is_shutdown_complete());

        m.shutdown();
        ShutdownCoordinator::instance().shutdown();  // leave the singleton reusable
    }

    // The begin -> checkpoint -> finalize worker used to be a detached thread, so nothing ever
    // waited for it: ThreadSanitizer reported ~Impl() destroying shutdown_cv while that worker
    // was still calling notify_all() on it. It is now a member joined by the destructor.
    //
    // That turns overwriting the handle into a hazard of its own -- assigning to a joinable
    // std::thread calls std::terminate -- so this pins that the singleton can serve several
    // shutdown rounds in a row (request, complete, reset, request again).
    TEST(PreemptionHandlerTest, ConsecutiveShutdownRoundsDoNotTerminate) {
        ShutdownCoordinator& c = ShutdownCoordinator::instance();
        ShutdownConfig cfg;
        cfg.checkpoint_on_shutdown = false;
        cfg.shutdown_timeout = std::chrono::seconds {5};
        c.initialize(cfg);

        for (int round = 0; round < 3; ++round) {
            EXPECT_NO_THROW(c.request_shutdown(15)) << "round " << round << " terminated instead of starting a worker";
            wait_complete(c);
            EXPECT_TRUE(c.is_shutdown_complete()) << "round " << round << " never reached Complete";
            c.shutdown();  // reset so the next round can start
        }
    }

}  // namespace nova::preemption::test
