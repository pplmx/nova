/**
 * @file comm_error_recovery_test.cpp
 * @brief Contracts for the communication error recovery layer
 *
 * include/cuda/comm/comm_error_recovery.h declares four singletons — a health
 * monitor, a retry handler with a circuit breaker, an NCCL error classifier and
 * a recovery coordinator — with no test coverage. Three defects lived here:
 *
 * - HealthMonitor, RetryHandler and CommErrorRecovery each hold their state
 *   behind `std::unique_ptr<Impl> impl_` and have a `= default` private
 *   constructor. Nothing allocated the Impl, so the first call through any
 *   member dereferenced a null pointer and killed the process. (ErrorClassifier
 *   is genuinely stateless and has no Impl.)
 *
 * - cuda_comm links only cudart and Threads, while comm_error_recovery.cpp
 *   gates its whole NCCL classification on NOVA_NCCL_ENABLED — a definition
 *   exported by cuda_nccl, which it did not link. The macro was therefore
 *   undefined in that translation unit, #if treated it as 0, and
 *   classify_from_nccl returned Unknown for every NCCL status.
 *
 * - Once the macro was visible, the NCCL branch did not compile: it named
 *   ncclTransportFailure and ncclTimeoutError, neither of which exists in any
 *   released nccl.h. The block had never been compiled before.
 */

#include <cuda/comm/comm_error_recovery.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>

namespace nova::comm::test {

    // --------------------------------------------------------- ErrorClassifier --

    TEST(ErrorClassifierTest, NcclStatusesAreClassifiedWhenNcclIsEnabled) {
        if (!NOVA_NCCL_ENABLED) {
            GTEST_SKIP() << "built without NCCL";
        }
        auto& classifier = ErrorClassifier::instance();

        EXPECT_EQ(classifier.classify_from_nccl(ncclSuccess), ErrorCategory::Unknown) << "success is not an error";
        EXPECT_EQ(classifier.classify_from_nccl(ncclUnhandledCudaError), ErrorCategory::NetworkError);
        EXPECT_EQ(classifier.classify_from_nccl(ncclRemoteError), ErrorCategory::NetworkError);
        EXPECT_EQ(classifier.classify_from_nccl(ncclSystemError), ErrorCategory::NetworkError);
        EXPECT_EQ(classifier.classify_from_nccl(ncclInternalError), ErrorCategory::InternalError);
        EXPECT_EQ(classifier.classify_from_nccl(ncclInvalidArgument), ErrorCategory::InvalidArgument);
        EXPECT_EQ(classifier.classify_from_nccl(ncclInvalidUsage), ErrorCategory::InvalidArgument);
        EXPECT_EQ(classifier.classify_from_nccl(ncclInProgress), ErrorCategory::Unknown);
    }

    // Regression: every one of these returned Unknown because the macro never
    // reached the translation unit. This asserts on a status that must map, so the
    // missing link cannot silently come back.
    TEST(ErrorClassifierTest, InvalidArgumentIsNotReportedAsUnknown) {
        if (!NOVA_NCCL_ENABLED) {
            GTEST_SKIP() << "built without NCCL";
        }
        EXPECT_NE(ErrorClassifier::instance().classify_from_nccl(ncclInvalidArgument), ErrorCategory::Unknown);
    }

    TEST(ErrorClassifierTest, PermanentAndTransientCategorisation) {
        auto& classifier = ErrorClassifier::instance();

        EXPECT_TRUE(classifier.is_permanent_error(ErrorCategory::InvalidArgument));
        EXPECT_TRUE(classifier.is_permanent_error(ErrorCategory::InternalError));
        EXPECT_FALSE(classifier.is_permanent_error(ErrorCategory::Timeout));

        EXPECT_TRUE(classifier.is_transient_error(ErrorCategory::Timeout));
        EXPECT_TRUE(classifier.is_transient_error(ErrorCategory::NetworkError));
        EXPECT_FALSE(classifier.is_transient_error(ErrorCategory::HardwareError));
    }

    TEST(ErrorClassifierTest, SeverityFollowsTheCategory) {
        auto& classifier = ErrorClassifier::instance();

        // A permanent category wins over everything else.
        EXPECT_EQ(classifier.classify_severity(ErrorCategory::InvalidArgument, ""), ErrorSeverity::Permanent);

        // A timeout is recoverable unless the details say it is intermittent.
        EXPECT_EQ(classifier.classify_severity(ErrorCategory::Timeout, "timed out"), ErrorSeverity::Recoverable);
        EXPECT_EQ(classifier.classify_severity(ErrorCategory::Timeout, "intermittent stall"), ErrorSeverity::Transient);

        EXPECT_EQ(classifier.classify_severity(ErrorCategory::NetworkError, ""), ErrorSeverity::Transient);
    }

    TEST(ErrorClassifierTest, EveryCategoryHasAMessage) {
        auto& classifier = ErrorClassifier::instance();
        const ErrorCategory all[] = {
            ErrorCategory::Timeout,
            ErrorCategory::NetworkError,
            ErrorCategory::HardwareError,
            ErrorCategory::InvalidArgument,
            ErrorCategory::InternalError,
            ErrorCategory::Unknown,
        };
        for (const ErrorCategory category : all) {
            EXPECT_FALSE(classifier.get_error_message(category).empty()) << "category " << static_cast<int>(category) << " has no message";
        }
    }

    TEST(CommErrorTest, RecoverabilityFollowsSeverity) {
        CommError error{};
        error.severity = ErrorSeverity::Recoverable;
        EXPECT_TRUE(error.is_recoverable());

        error.severity = ErrorSeverity::Transient;
        EXPECT_TRUE(error.is_recoverable());

        error.severity = ErrorSeverity::Permanent;
        EXPECT_FALSE(error.is_recoverable());
    }

    // ------------------------------------------------------------ RetryHandler --

    TEST(RetryHandlerTest, DefaultConfigIsSane) {
        auto& handler = RetryHandler::instance();

        RetryHandler::RetryConfig config;
        config.max_attempts = 4;
        config.initial_delay = std::chrono::milliseconds(1);
        config.max_delay = std::chrono::milliseconds(4);
        config.use_jitter = false;
        handler.set_config(config);

        const auto readback = handler.get_config();
        EXPECT_EQ(readback.max_attempts, 4);
        EXPECT_EQ(readback.initial_delay.count(), 1);
        EXPECT_FALSE(readback.use_jitter);

        // Anything the file itself reads before this point would have crashed on a
        // null impl_; reading it back proves the Impl exists.
        EXPECT_FALSE(handler.is_circuit_open());
    }

    TEST(RetryHandlerTest, ExecuteReturnsTheFunctionResult) {
        auto& handler = RetryHandler::instance();
        handler.reset_circuit_breaker();

        int calls = 0;
        const int result = handler.execute_with_retry(
            [&calls] {
                ++calls;
                return 42;
            },
            "test-operation");

        EXPECT_EQ(result, 42);
        EXPECT_EQ(calls, 1) << "a successful call must not be retried";
    }

    TEST(RetryHandlerTest, TransientFailureIsRetriedThenSucceeds) {
        auto& handler = RetryHandler::instance();
        RetryHandler::RetryConfig config;
        config.max_attempts = 3;
        config.initial_delay = std::chrono::milliseconds(1);
        config.max_delay = std::chrono::milliseconds(1);
        config.use_jitter = false;
        handler.set_config(config);
        handler.reset_circuit_breaker();

        int attempts = 0;
        const int result = handler.execute_with_retry(
            [&attempts] {
                if (++attempts < 3) {
                    throw std::runtime_error("transient");
                }
                return 7;
            },
            "retry-me");

        EXPECT_EQ(result, 7);
        EXPECT_EQ(attempts, 3);
    }

    TEST(RetryHandlerTest, ExhaustedRetriesPropagateTheLastError) {
        auto& handler = RetryHandler::instance();
        RetryHandler::RetryConfig config;
        config.max_attempts = 2;
        config.initial_delay = std::chrono::milliseconds(1);
        config.max_delay = std::chrono::milliseconds(1);
        config.use_jitter = false;
        handler.set_config(config);
        handler.reset_circuit_breaker();

        int attempts = 0;
        EXPECT_THROW(handler.execute_with_retry(
                         [&attempts]() -> int {
                             ++attempts;
                             throw std::runtime_error("always fails");
                         },
                         "doomed"),
                     std::runtime_error);
        EXPECT_EQ(attempts, 2) << "max_attempts is the total number of tries";

        handler.reset_circuit_breaker();
    }

    // ----------------------------------------------------------- HealthMonitor --

    // Every call here dereferenced a null impl_ before the singleton allocated it.
    TEST(CommHealthMonitorTest, LifecycleAndRegistrationDoNotCrash) {
        auto& monitor = HealthMonitor::instance();

        monitor.set_timeout_threshold(std::chrono::seconds(5));
        monitor.set_check_interval(std::chrono::milliseconds(10));
        monitor.set_stall_confirmations(2);

        // An unregistered communicator has no status, which is a well-defined
        // answer rather than a crash.
        EXPECT_FALSE(monitor.get_status(nullptr).has_value());

        int comm_marker = 0;
        monitor.register_comm(&comm_marker, nullptr);
        const auto status = monitor.get_status(&comm_marker);
        ASSERT_TRUE(status.has_value());
        EXPECT_EQ(status->comm_id, &comm_marker);
        EXPECT_TRUE(status->is_healthy);

        monitor.unregister_comm(&comm_marker);
        EXPECT_FALSE(monitor.get_status(&comm_marker).has_value());
    }

    TEST(CommHealthMonitorTest, StartAndStopAreIdempotent) {
        auto& monitor = HealthMonitor::instance();
        monitor.set_check_interval(std::chrono::milliseconds(5));
        monitor.set_error_callback(nullptr);

        monitor.start();
        EXPECT_TRUE(monitor.is_running());
        monitor.stop();
        EXPECT_FALSE(monitor.is_running());
        monitor.stop();
        EXPECT_FALSE(monitor.is_running()) << "stopping twice must be safe";
    }

    // --------------------------------------------------------- CommErrorRecovery --

    TEST(CommErrorRecoveryTest, ShouldRetryFollowsRecoverability) {
        auto& recovery = CommErrorRecovery::instance();

        CommError transient_error{};
        transient_error.category = ErrorCategory::NetworkError;
        transient_error.severity = ErrorSeverity::Transient;
        EXPECT_TRUE(recovery.should_retry(transient_error));

        CommError permanent_error{};
        permanent_error.category = ErrorCategory::InvalidArgument;
        permanent_error.severity = ErrorSeverity::Permanent;
        EXPECT_FALSE(recovery.should_retry(permanent_error)) << "a permanent error must not be retried";
    }

    TEST(CommErrorRecoveryTest, InitialiseAndShutdownAreSafe) {
        auto& recovery = CommErrorRecovery::instance();

        // initialize() wires the health monitor, the retry handler and the
        // classifier together; none of those pointers existed before.
        EXPECT_NO_THROW(recovery.initialize(nullptr));
        EXPECT_NO_THROW(recovery.shutdown());
        EXPECT_NO_THROW(recovery.shutdown()) << "shutdown must be idempotent";
    }

    TEST(CommErrorRecoveryTest, OnCommErrorIsAccepted) {
        auto& recovery = CommErrorRecovery::instance();

        CommError error{};
        error.category = ErrorCategory::Timeout;
        error.severity = ErrorSeverity::Recoverable;
        error.message = "collective timed out";
        error.device_id = 0;

        EXPECT_NO_THROW(recovery.on_comm_error(error));
    }

    TEST(CommErrorRecoveryTest, TimeoutThresholdIsSettable) {
        auto& recovery = CommErrorRecovery::instance();
        EXPECT_NO_THROW(recovery.set_timeout_threshold(std::chrono::seconds(30)));
    }

}  // namespace nova::comm::test
