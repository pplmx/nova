#include <gtest/gtest.h>
#include <thread>
#include "cuda/error/retry.hpp"

using namespace nova::error;

class RetryTest : public ::testing::Test {};

TEST_F(RetryTest, ExponentialBackoffCalculation) {
    auto delay = calculate_backoff(1, std::chrono::milliseconds{100}, 2.0, std::chrono::seconds{30});
    EXPECT_EQ(delay.count(), 100);

    delay = calculate_backoff(2, std::chrono::milliseconds{100}, 2.0, std::chrono::seconds{30});
    EXPECT_EQ(delay.count(), 200);

    delay = calculate_backoff(3, std::chrono::milliseconds{100}, 2.0, std::chrono::seconds{30});
    EXPECT_EQ(delay.count(), 400);
}

TEST_F(RetryTest, BackoffRespectsMaxDelay) {
    auto delay = calculate_backoff(10, std::chrono::milliseconds{100}, 2.0, std::chrono::seconds{1});
    EXPECT_EQ(delay.count(), 1000);
}

TEST_F(RetryTest, FullJitterReturnsValueInRange) {
    auto delay = full_jitter(std::chrono::milliseconds{100});
    EXPECT_GE(delay.count(), 0);
    EXPECT_LE(delay.count(), 100);
}

TEST_F(RetryTest, CircuitBreakerStartsClosed) {
    circuit_breaker cb({});
    EXPECT_EQ(cb.state(), circuit_state::closed);
    EXPECT_TRUE(cb.allow_request());
}

TEST_F(RetryTest, CircuitBreakerOpensAfterThresholdFailures) {
    circuit_breaker_config config;
    config.failure_threshold = 3;
    circuit_breaker cb(config);

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(cb.allow_request());
        cb.record_failure();
    }

    EXPECT_EQ(cb.state(), circuit_state::open);
    EXPECT_FALSE(cb.allow_request());
}

TEST_F(RetryTest, CircuitBreakerTransitionsToHalfOpenAfterTimeout) {
    circuit_breaker_config config;
    config.failure_threshold = 1;
    config.reset_timeout = std::chrono::milliseconds{10};
    circuit_breaker cb(config);

    cb.record_failure();
    EXPECT_EQ(cb.state(), circuit_state::open);

    std::this_thread::sleep_for(std::chrono::milliseconds{15});
    EXPECT_TRUE(cb.allow_request());
    EXPECT_EQ(cb.state(), circuit_state::half_open);
}

TEST_F(RetryTest, CircuitBreakerClosesAfterSuccessesInHalfOpen) {
    circuit_breaker_config config;
    config.failure_threshold = 1;
    config.half_open_success_threshold = 2;
    circuit_breaker cb(config);

    cb.record_failure();
    std::this_thread::sleep_for(std::chrono::milliseconds{15});
    cb.allow_request();

    EXPECT_EQ(cb.state(), circuit_state::half_open);
    cb.record_success();
    cb.record_success();

    EXPECT_EQ(cb.state(), circuit_state::closed);
}

TEST_F(RetryTest, RetryExecutorSucceedsOnFirstAttempt) {
    retry_config config;
    config.max_attempts = 3;
    config.jitter_enabled = false;

    retry_executor executor(config);

    bool called = false;
    executor.execute([&]() {
        called = true;
        return 42;
    });

    EXPECT_TRUE(called);
    EXPECT_TRUE(executor.was_successful());
    EXPECT_EQ(executor.attempt_count(), 1);
}

TEST_F(RetryTest, RetryExecutorRetriesOnFailure) {
    retry_config config;
    config.max_attempts = 3;
    config.jitter_enabled = false;
    config.base_delay = std::chrono::milliseconds{10};

    retry_executor executor(config);

    int attempts = 0;
    executor.execute([&]() {
        ++attempts;
        if (attempts < 2) {
            throw std::runtime_error("Transient error");
        }
        return 42;
    });

    EXPECT_EQ(executor.attempt_count(), 2);
    EXPECT_TRUE(executor.was_successful());
}
