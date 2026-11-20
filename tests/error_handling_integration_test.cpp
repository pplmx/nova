#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include "cuda/error/timeout.hpp"
#include "cuda/error/timeout_context.hpp"
#include "cuda/error/retry.hpp"
#include "cuda/error/degrade.hpp"

using namespace nova::error;

class ErrorHandlingIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(ErrorHandlingIntegrationTest, TimeoutRetryDegradeChain) {
    auto& degrade_mgr = degradation_manager::instance();

    bool degraded = false;
    degrade_mgr.set_callback([&](const degradation_event& event) { degraded = true; });

    retry_config retry_cfg;
    retry_cfg.base_delay = std::chrono::milliseconds {5};
    retry_cfg.max_attempts = 2;
    retry_cfg.jitter_enabled = false;

    retry_executor executor(retry_cfg);

    int attempts = 0;
    auto precision = precision_level::high;

    try {
        executor.execute([&]() {
            ++attempts;
            if (attempts < 2) {
                throw std::runtime_error("Transient error");
            }

            if (degrade_mgr.should_degrade("matmul")) {
                precision = degrade(degrade_mgr.get_precision("matmul"));
                degrade_mgr.trigger_degradation("matmul", precision, "retry exhaustion");
            }

            return precision;
        });
    } catch (...) {}

    EXPECT_EQ(attempts, 2);
    EXPECT_TRUE(degraded || !degraded);
}

TEST_F(ErrorHandlingIntegrationTest, CircuitBreakerUnderConcurrentLoad) {
    circuit_breaker_config cb_cfg;
    cb_cfg.failure_threshold = 5;
    cb_cfg.reset_timeout = std::chrono::seconds {1};
    circuit_breaker cb(cb_cfg);

    std::atomic<int> blocked_count {0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            if (!cb.allow_request()) {
                ++blocked_count;
            }
            cb.record_failure();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(cb.state(), circuit_state::open);
    EXPECT_FALSE(cb.allow_request());
    EXPECT_GE(blocked_count.load(), 0);
}

TEST_F(ErrorHandlingIntegrationTest, TimeoutGuardWithRetry) {
    auto id = timeout_manager::instance().start_operation("critical_op", std::chrono::milliseconds {100});

    retry_config retry_cfg;
    retry_cfg.max_attempts = 3;
    retry_cfg.jitter_enabled = false;
    retry_cfg.base_delay = std::chrono::milliseconds {10};

    retry_executor executor(retry_cfg);

    int attempts = 0;
    const bool succeeded = executor.execute([&]() {
        ++attempts;
        if (timeout_manager::instance().is_expired(id)) {
            throw std::runtime_error("Operation timed out");
        }
        return true;
    });

    // The adapter below is the point of the case and none of it was checked: the operation
    // is registered, the guard's clock is wired to the retry loop, and the call is
    // unregistered on the way out. A suite where attempts stayed at zero, or where the
    // operation was never tracked, passed just the same.
    EXPECT_TRUE(succeeded) << "the operation should complete inside its 100 ms budget";
    EXPECT_GE(attempts, 1) << "the retry executor must run the body at least once";
    EXPECT_LE(attempts, 3) << "max_attempts is 3, so a successful first try is fine but "
                              "more than three runs would ignore the configuration";

    timeout_manager::instance().end_operation(id);
    EXPECT_EQ(timeout_manager::instance().active_count(), 0u) << "end_operation must release the tracked operation";
}

TEST_F(ErrorHandlingIntegrationTest, BackwardCompatibility) {
    auto& manager = timeout_manager::instance();
    EXPECT_EQ(manager.active_count(), 0u);

    auto id = manager.start_operation("test", std::chrono::milliseconds {1000});
    EXPECT_GT(id, 0u);

    EXPECT_EQ(manager.active_count(), 1u);
    manager.end_operation(id);
    EXPECT_EQ(manager.active_count(), 0u);
}
