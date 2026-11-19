/**
 * @file memory_error_handler_test.cpp
 * @brief Contracts for the memory error detection and degradation layer
 *
 * include/cuda/memory_error/memory_error_handler.h declares three singletons
 * (DeviceHealthMonitor, MemoryErrorHandler, DegradationManager), an error
 * classifier and a degradation state machine, with no test coverage. Three
 * defects lived here:
 *
 * - All three singletons have `= default` private constructors and hold their
 *   state behind `std::unique_ptr<Impl> impl_`, which nothing ever allocated.
 *   The first call through any member dereferenced a null pointer and crashed,
 *   so the whole layer was unusable.
 *
 * - CudaErrorDetector::create_error was declared but never defined: calling it
 *   failed to link.
 *
 * These tests exercise every member that would have crashed, and pin the
 * classification rules that the rest of the layer depends on.
 */

#include <cuda/memory_error/memory_error_handler.h>

#include <gtest/gtest.h>

#include <string>

namespace nova::memory::test {

    // ------------------------------------------------------- CudaErrorDetector --

    TEST(CudaErrorDetectorTest, OutOfMemoryIsAnAllocationFailure) {
        auto& detector = CudaErrorDetector::instance();

        EXPECT_TRUE(detector.is_memory_related(cudaErrorMemoryAllocation));
        EXPECT_TRUE(detector.is_allocation_failure(cudaErrorMemoryAllocation));
        EXPECT_EQ(detector.classify_error(cudaErrorMemoryAllocation), MemoryErrorCategory::AllocationFailure);
    }

    TEST(CudaErrorDetectorTest, PeerAccessErrorsAreCategorised) {
        auto& detector = CudaErrorDetector::instance();

        EXPECT_EQ(detector.classify_error(cudaErrorPeerAccessNotEnabled), MemoryErrorCategory::PeerAccessError);
        EXPECT_EQ(detector.classify_error(cudaErrorPeerAccessAlreadyEnabled), MemoryErrorCategory::PeerAccessError);
    }

    TEST(CudaErrorDetectorTest, UnrelatedErrorsAreStillCudaErrors) {
        auto& detector = CudaErrorDetector::instance();

        EXPECT_EQ(detector.classify_error(cudaErrorInvalidValue), MemoryErrorCategory::CudaError);
        EXPECT_FALSE(detector.is_memory_related(cudaErrorInvalidValue));
        EXPECT_FALSE(detector.is_allocation_failure(cudaErrorInvalidValue));
    }

    TEST(CudaErrorDetectorTest, SeverityEscalatesForOutOfMemory) {
        auto& detector = CudaErrorDetector::instance();

        // An out-of-memory allocation failure is Critical: the caller has to free
        // something before it can make progress. Other allocation failures are
        // Recoverable.
        EXPECT_EQ(detector.classify_severity(MemoryErrorCategory::AllocationFailure, cudaErrorMemoryAllocation), MemoryErrorSeverity::Critical);

        // An invalid argument is a Warning — the program is fine, the call was not.
        EXPECT_EQ(detector.classify_severity(MemoryErrorCategory::CudaError, cudaErrorInvalidValue), MemoryErrorSeverity::Warning);

        // Corruption is always Fatal regardless of the CUDA code.
        EXPECT_EQ(detector.classify_severity(MemoryErrorCategory::MemoryCorruption, cudaSuccess), MemoryErrorSeverity::Fatal);
    }

    TEST(CudaErrorDetectorTest, DescribeMatchesTheRuntimeString) {
        auto& detector = CudaErrorDetector::instance();

        EXPECT_EQ(detector.get_error_description(cudaErrorMemoryAllocation), cudaGetErrorString(cudaErrorMemoryAllocation));
        EXPECT_FALSE(detector.get_error_description(cudaErrorInvalidValue).empty());
    }

    // create_error had no definition at all, so this test would not even link
    // against the old code.
    TEST(CudaErrorDetectorTest, CreateErrorFillsCategorySeverityAndMessage) {
        auto& detector = CudaErrorDetector::instance();

        const MemoryError error = detector.create_error(cudaErrorMemoryAllocation, 1, 4096);

        EXPECT_EQ(error.cuda_error, cudaErrorMemoryAllocation);
        EXPECT_EQ(error.device_id, 1);
        EXPECT_EQ(error.attempted_size, 4096u);
        EXPECT_EQ(error.category, MemoryErrorCategory::AllocationFailure);
        EXPECT_EQ(error.severity, MemoryErrorSeverity::Critical);
        EXPECT_EQ(error.message, cudaGetErrorString(cudaErrorMemoryAllocation));
        EXPECT_FALSE(error.is_recoverable());
    }

    TEST(MemoryErrorTest, RecoverabilityFollowsSeverity) {
        MemoryError error {};
        error.severity = MemoryErrorSeverity::Warning;
        EXPECT_TRUE(error.is_recoverable());

        error.severity = MemoryErrorSeverity::Recoverable;
        EXPECT_TRUE(error.is_recoverable());

        error.severity = MemoryErrorSeverity::Critical;
        EXPECT_FALSE(error.is_recoverable());

        error.severity = MemoryErrorSeverity::Fatal;
        EXPECT_FALSE(error.is_recoverable());
    }

    TEST(MemoryErrorTest, ErrorStringComesFromTheRuntime) {
        MemoryError error {};
        error.cuda_error = cudaErrorMemoryAllocation;
        EXPECT_EQ(error.get_cuda_error_string(), cudaGetErrorString(cudaErrorMemoryAllocation));
    }

    // ------------------------------------------------------ DegradationManager --
    //
    // Every one of these members dereferenced a null impl_ before the singleton
    // allocated it, so any call aborted the process.

    TEST(DegradationManagerTest, StartsAtNominal) {
        EXPECT_EQ(DegradationManager::instance().get_current_level(), DegradationManager::DegradationLevel::Nominal);
    }

    TEST(DegradationManagerTest, ApplyingALevelIsObservable) {
        auto& manager = DegradationManager::instance();
        const auto original = manager.get_current_level();

        manager.apply_degradation(DegradationManager::DegradationLevel::ReducedBatch);
        EXPECT_EQ(manager.get_current_level(), DegradationManager::DegradationLevel::ReducedBatch);

        manager.apply_degradation(original);
    }

    TEST(DegradationManagerTest, OperationSpecificDegradationSetsMatchingLevel) {
        auto& manager = DegradationManager::instance();

        manager.reduce_tensor_parallelism(4);
        EXPECT_EQ(manager.get_current_level(), DegradationManager::DegradationLevel::ReducedTP);

        manager.increase_batch_size_reduction(0.5f);
        EXPECT_EQ(manager.get_current_level(), DegradationManager::DegradationLevel::ReducedBatch);

        manager.trigger_cpu_fallback();
        EXPECT_EQ(manager.get_current_level(), DegradationManager::DegradationLevel::CPUFallback);

        manager.apply_degradation(DegradationManager::DegradationLevel::Nominal);
    }

    TEST(DegradationManagerTest, RecoveryIsAllowedExceptFromAbortAndFallback) {
        auto& manager = DegradationManager::instance();

        EXPECT_TRUE(manager.can_recover(DegradationManager::DegradationLevel::Nominal));
        EXPECT_TRUE(manager.can_recover(DegradationManager::DegradationLevel::ReducedTP));
        EXPECT_TRUE(manager.can_recover(DegradationManager::DegradationLevel::ReducedBatch));
        EXPECT_FALSE(manager.can_recover(DegradationManager::DegradationLevel::CPUFallback));
        EXPECT_FALSE(manager.can_recover(DegradationManager::DegradationLevel::Abort));
    }

    TEST(DegradationManagerTest, AttemptRecoveryReturnsToNominal) {
        auto& manager = DegradationManager::instance();

        manager.trigger_cpu_fallback();
        ASSERT_FALSE(manager.can_recover(manager.get_current_level()));

        manager.attempt_recovery();
        EXPECT_EQ(manager.get_current_level(), DegradationManager::DegradationLevel::Nominal);
    }

    // ----------------------------------------------------- MemoryErrorHandler --

    TEST(MemoryErrorHandlerTest, HandlerAcceptsErrorsAndTracksTelemetry) {
        auto& handler = MemoryErrorHandler::instance();
        handler.reset_telemetry();

        auto& detector = CudaErrorDetector::instance();
        handler.handle_error(detector.create_error(cudaErrorMemoryAllocation, 0, 1024));

        const auto telemetry = handler.get_telemetry();
        EXPECT_EQ(telemetry.total_errors, 1);
        EXPECT_GE(telemetry.critical_errors, 1) << "an out-of-memory error is classified Critical";

        handler.reset_telemetry();
        EXPECT_EQ(handler.get_telemetry().total_errors, 0);
    }

    TEST(MemoryErrorHandlerTest, CallbacksFireForHandledErrors) {
        auto& handler = MemoryErrorHandler::instance();
        handler.reset_telemetry();

        std::string seen;
        handler.register_error_callback([&seen](const MemoryError& error) { seen = error.message; });

        handler.handle_error(CudaErrorDetector::instance().create_error(cudaErrorInvalidValue, 0, 0));

        EXPECT_EQ(seen, cudaGetErrorString(cudaErrorInvalidValue));

        // Replace the callback so it does not observe later tests' errors.
        handler.register_error_callback(nullptr);
        handler.reset_telemetry();
    }

    TEST(MemoryErrorHandlerTest, TensorParallelismRecommendationIsClamped) {
        auto& handler = MemoryErrorHandler::instance();

        handler.set_current_tp_degree(8);
        EXPECT_GE(handler.get_recommended_tp_degree(), 1);

        handler.set_current_tp_degree(1);
        EXPECT_EQ(handler.get_recommended_tp_degree(), 1) << "tp degree cannot go below 1";
    }

    TEST(MemoryErrorHandlerTest, OutOfMemoryAsksForLessParallelismButNotCpuYet) {
        auto& handler = MemoryErrorHandler::instance();
        handler.reset_telemetry();

        const MemoryError oom = CudaErrorDetector::instance().create_error(cudaErrorMemoryAllocation, 0, 1u << 30);

        // A critical allocation failure must immediately ask for reduced
        // parallelism — that is the cheap mitigation.
        EXPECT_TRUE(handler.should_reduce_parallelism(oom));

        // Falling back to the CPU is the escalation, not the first response: it is
        // reserved for a fatal error, memory corruption, or repeated degradation.
        EXPECT_FALSE(handler.should_fallback_to_cpu(oom)) << "a single OOM should be handled by reducing parallelism first";
    }

    TEST(MemoryErrorHandlerTest, RepeatedDegradationEscalatesToCpuFallback) {
        // The escalation rule: after three degradation attempts the cheap
        // mitigations are exhausted and the CPU path is offered.
        auto& handler = MemoryErrorHandler::instance();
        handler.reset_telemetry();

        const MemoryError oom = CudaErrorDetector::instance().create_error(cudaErrorMemoryAllocation, 0, 1024);

        // The escalation threshold is three degradations.
        for (int i = 0; i < 2; ++i) {
            EXPECT_FALSE(handler.should_fallback_to_cpu(oom)) << "degradation " << i + 1 << " of 3 should still be recoverable in place";
            handler.handle_error(oom);
        }
        handler.handle_error(oom);  // third: the escalation point
        EXPECT_TRUE(handler.should_fallback_to_cpu(oom)) << "after three degradations the CPU fallback should be offered";

        handler.reset_telemetry();
    }

    TEST(MemoryErrorHandlerTest, FatalErrorGoesStraightToCpuFallback) {
        auto& handler = MemoryErrorHandler::instance();
        handler.reset_telemetry();

        MemoryError corruption {};
        corruption.category = MemoryErrorCategory::MemoryCorruption;
        corruption.severity = MemoryErrorSeverity::Fatal;
        corruption.cuda_error = cudaSuccess;

        EXPECT_TRUE(handler.should_fallback_to_cpu(corruption)) << "memory corruption is not recoverable on the device";

        handler.reset_telemetry();
    }

    TEST(MemoryErrorHandlerTest, InvalidArgumentDoesNotTriggerDegradation) {
        auto& handler = MemoryErrorHandler::instance();
        const MemoryError bad_arg = CudaErrorDetector::instance().create_error(cudaErrorInvalidValue, 0, 0);

        EXPECT_FALSE(handler.should_reduce_parallelism(bad_arg));
        EXPECT_FALSE(handler.should_fallback_to_cpu(bad_arg));
    }

    // ---------------------------------------------------- DeviceHealthMonitor --

    TEST(DeviceHealthMonitorTest, ThresholdsAreSettableAndQueriesDoNotCrash) {
        auto& monitor = DeviceHealthMonitor::instance();

        // Both of these used to dereference a null impl_.
        monitor.set_check_interval(std::chrono::milliseconds(250));
        monitor.set_memory_threshold(0.8f, 0.95f);

        const auto health = monitor.get_health(0);
        EXPECT_EQ(health.device_id, 0);
        EXPECT_FALSE(health.status_message.empty());

        // is_device_healthy on an unpopulated cache reports the conservative state
        // rather than throwing.
        (void)monitor.is_device_healthy(0);
    }

    TEST(DeviceHealthMonitorTest, HealthCallbacksCanBeReplacedAndCleared) {
        auto& monitor = DeviceHealthMonitor::instance();

        monitor.set_health_callback([](const DeviceHealthMonitor::DeviceHealth&) {});
        monitor.set_health_callback(nullptr);

        // Monitoring is the only path that invokes the callback; starting and
        // stopping it must not crash or hang.
        monitor.start_monitoring();
        monitor.stop_monitoring();
        EXPECT_NO_THROW(monitor.stop_monitoring()) << "stopping twice must be safe";
    }

    TEST(MemoryErrorTest, EccLimitationNoteExplainsTheGap) {
        // The layer cannot read ECC counters through the CUDA public API; the note
        // is the documented statement of that limitation.
        const std::string note = get_ecc_limitations_note();
        EXPECT_NE(note.find("ECC"), std::string::npos);
        EXPECT_NE(note.find("NVML"), std::string::npos);
    }

}  // namespace nova::memory::test
