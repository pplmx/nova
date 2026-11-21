#pragma once

#include <string>
#include <functional>
#include <atomic>
#include <chrono>
#include <memory>
#include <cuda_runtime.h>

namespace nova::memory {

    enum class MemoryErrorCategory { CudaError, AllocationFailure, MemoryCorruption, ECCError, PeerAccessError, Unknown };

    enum class MemoryErrorSeverity { Warning, Recoverable, Critical, Fatal };

    struct MemoryError {
        MemoryErrorCategory category;
        MemoryErrorSeverity severity;
        cudaError_t cuda_error;
        int device_id;
        size_t attempted_size;
        std::string message;
        std::chrono::steady_clock::time_point timestamp;

        std::string get_cuda_error_string() const {
            const char* err_str = cudaGetErrorString(cuda_error);
            return err_str ? std::string(err_str) : "Unknown CUDA error";
        }

        bool is_recoverable() const { return severity == MemoryErrorSeverity::Warning || severity == MemoryErrorSeverity::Recoverable; }
    };

    class DeviceHealthMonitor {
    public:
        static DeviceHealthMonitor& instance();

        struct DeviceHealth {
            int device_id;
            bool is_healthy;
            size_t memory_used;
            size_t memory_total;
            float utilization;
            int ecc_errors;
            std::string status_message;
        };

        void start_monitoring();
        void stop_monitoring();

        /**
         * @brief Last measured health for a device
         *
         * Returns a default-constructed record with status_message "Unknown" when the monitor
         * has not measured that device -- including any id outside the device count.
         */
        DeviceHealth get_health(int device_id);

        /**
         * @brief Whether the monitor has measured this device as healthy
         *
         * False when nothing is known about the id, because "not measured" is not "healthy":
         * a caller gating work on this must not proceed on a device the monitor has never
         * seen. This used to return true for every unknown id, contradicting get_health(),
         * which reports the same id as "Unknown".
         */
        bool is_device_healthy(int device_id) const;

        void set_check_interval(std::chrono::milliseconds interval);
        void set_memory_threshold(float warning_threshold, float critical_threshold);

        using HealthCallback = std::function<void(const DeviceHealth&)>;
        void set_health_callback(HealthCallback callback);

    private:
        DeviceHealthMonitor() = default;
        ~DeviceHealthMonitor();

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class MemoryErrorHandler {
    public:
        static MemoryErrorHandler& instance();

        void initialize();
        void shutdown();

        void handle_error(const MemoryError& error);

        void register_error_callback(std::function<void(const MemoryError&)> callback);

        bool should_reduce_parallelism(const MemoryError& error);
        bool should_fallback_to_cpu(const MemoryError& error);

        int get_recommended_tp_degree() const;
        void set_current_tp_degree(int degree);

        struct Telemetry {
            int total_errors;
            int critical_errors;
            int recoverable_errors;
            int reductions_triggered;
            size_t peak_memory_usage;
            std::chrono::seconds uptime;
        };

        Telemetry get_telemetry() const;
        void reset_telemetry();

    private:
        MemoryErrorHandler() = default;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class CudaErrorDetector {
    public:
        static CudaErrorDetector& instance();

        MemoryErrorCategory classify_error(cudaError_t error);
        MemoryErrorSeverity classify_severity(MemoryErrorCategory category, cudaError_t error);

        std::string get_error_description(cudaError_t error);

        bool is_memory_related(cudaError_t error);
        bool is_allocation_failure(cudaError_t error);

        MemoryError create_error(cudaError_t error, int device_id, size_t attempted_size = 0);

    private:
        CudaErrorDetector() = default;

        bool is_out_of_memory(cudaError_t error);
        bool is_peer_access_error(cudaError_t error);
        bool is_invalid_value(cudaError_t error);
    };

    class DegradationManager {
    public:
        static DegradationManager& instance();

        enum class DegradationLevel { Nominal, ReducedTP, ReducedBatch, CPUFallback, Abort };

        void apply_degradation(DegradationLevel level);
        DegradationLevel get_current_level() const;

        void reduce_tensor_parallelism(int device_count);
        void increase_batch_size_reduction(float factor);
        void trigger_cpu_fallback();

        bool can_recover(DegradationLevel level) const;
        void attempt_recovery();

    private:
        DegradationManager() = default;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    MemoryError make_memory_error(cudaError_t error, int device_id, size_t attempted_size = 0, MemoryErrorCategory category = MemoryErrorCategory::CudaError);

}  // namespace nova::memory

namespace nova::memory {
    inline std::string get_ecc_limitations_note() {
        return "ECC error detection requires nvidia-smi or NVML integration. "
               "CUDA public API does not expose ECC error counts. "
               "For ECC monitoring, consider using NVML APIs or nvidia-smi queries.";
    }
}  // namespace nova::memory
