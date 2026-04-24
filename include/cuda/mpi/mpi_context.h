#pragma once

/**
 * @file mpi_context.h
 * @brief MPI context for multi-node rank discovery and NCCL bootstrapping
 *
 * Provides centralized MPI management with RAII semantics for lifecycle
 * control and rank/node discovery for multi-node training.
 *
 * @example Basic usage:
 * @code
 * auto& mpi = cuda::mpi::MpiContext::instance();
 * mpi.initialize(&argc, &argv);
 *
 * if (mpi.world_rank() == 0) {
 *     printf("Running on %d nodes\n", mpi.world_size());
 * }
 * @endcode
 *
 * @example With config:
 * @code
 * cuda::mpi::MpiConfig config;
 * config.thread_level = MPI_THREAD_SERIALIZED;
 * config.timeout_ms = 60000;
 *
 * auto& mpi = cuda::mpi::MpiContext::instance();
 * mpi.initialize(config);
 * @endcode
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if NOVA_MPI_ENABLED
#include <mpi.h>
#endif

namespace cuda::mpi {

/**
 * @class MpiException
 * @brief Exception thrown on MPI errors
 */
class MpiException : public std::runtime_error {
public:
    MpiException(int error_code, const char* expr, const char* file, int line);
    [[nodiscard]] int error_code() const noexcept { return error_code_; }

private:
    int error_code_;
};

/**
 * @def MPI_CHECK(call)
 * @brief Check MPI call result and throw on error
 */
#if NOVA_MPI_ENABLED
#define MPI_CHECK(call)                                                          \
    do {                                                                          \
        int _err = call;                                                          \
        if (_err != MPI_SUCCESS) {                                                \
            throw cuda::mpi::MpiException(_err, #call, __FILE__, __LINE__);       \
        }                                                                         \
    } while (0)
#else
#define MPI_CHECK(call) ((void)0)
#endif

/**
 * @struct MpiConfig
 * @brief Configuration options for MpiContext initialization
 */
struct MpiConfig {
    int timeout_ms = 30000;
    int thread_level = 3;  // MPI_THREAD_MULTIPLE
    std::optional<int> local_device_limit;
    bool debug = false;
};

/**
 * @class MpiContext
 * @brief MPI context with singleton pattern and RAII lifecycle
 *
 * Manages MPI initialization/finalization and provides rank discovery.
 * Thread-safe singleton ensures single initialization across the program.
 */
class MpiContext {
public:
    static MpiContext& instance();

    MpiContext(const MpiContext&) = delete;
    MpiContext& operator=(const MpiContext&) = delete;

    void initialize(const MpiConfig& config = {});
    void initialize(int* argc, char*** argv, const MpiConfig& config = {});
    void finalize();

    [[nodiscard]] int world_rank() const { return world_rank_; }
    [[nodiscard]] int world_size() const { return world_size_; }
    [[nodiscard]] int local_rank() const { return local_rank_; }
    [[nodiscard]] int local_size() const { return local_size_; }
    [[nodiscard]] int node_id() const { return node_id_; }
    [[nodiscard]] bool is_main_process() const { return world_rank_ == 0; }
    [[nodiscard]] bool is_main_node() const { return local_rank_ == 0; }
    [[nodiscard]] bool initialized() const { return initialized_; }
    [[nodiscard]] bool has_mpi() const { return initialized_ && world_size_ > 1; }

    [[nodiscard]] int get_local_device_id() const;

    [[nodiscard]] const std::string& hostname() const { return hostname_; }

private:
    MpiContext() = default;
    ~MpiContext();

    void compute_local_rank();
    void compute_node_id();
    void gather_hostnames();

    int world_rank_ = 0;
    int world_size_ = 1;
    int local_rank_ = 0;
    int local_size_ = 1;
    int node_id_ = 0;
    bool initialized_ = false;
    std::string hostname_;
    MpiConfig config_;
};

void validate_mpi_environment();

}  // namespace cuda::mpi
