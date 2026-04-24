#include "cuda/mpi/mpi_context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace cuda::mpi {

namespace {
std::mutex init_mutex;
}

MpiException::MpiException(int error_code, const char* expr,
                           const char* file, int line)
    : std::runtime_error("MPI error " + std::to_string(error_code) +
                         " in " + std::string(expr) + " at " +
                         std::string(file) + ":" + std::to_string(line)),
      error_code_(error_code) {}

MpiContext& MpiContext::instance() {
    static MpiContext ctx;
    return ctx;
}

MpiContext::~MpiContext() {
    if (initialized_) {
        finalize();
    }
}

void MpiContext::initialize(const MpiConfig& config) {
    std::lock_guard<std::mutex> lock(init_mutex);

    if (initialized_) {
        return;
    }

#if NOVA_MPI_ENABLED
    int provided;
    const int required = config.thread_level;

    MPI_CHECK(MPI_Init_thread(nullptr, nullptr, required, &provided));

    if (provided < required && config.debug) {
        std::fprintf(stderr, "MPI: Thread level %d not available, got %d\n",
                     required, provided);
    }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size_));

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_CHECK(MPI_Get_processor_name(name, &len));
    hostname_ = std::string(name, len);

    config_ = config;
    compute_local_rank();
    compute_node_id();
#else
    (void)config;
    world_rank_ = 0;
    world_size_ = 1;
    local_rank_ = 0;
    local_size_ = 1;
    node_id_ = 0;
#endif

    initialized_ = true;
}

void MpiContext::initialize(int* argc, char*** argv, const MpiConfig& config) {
    std::lock_guard<std::mutex> lock(init_mutex);

    if (initialized_) {
        return;
    }

#if NOVA_MPI_ENABLED
    int provided;
    const int required = config.thread_level;

    MPI_CHECK(MPI_Init_thread(argc, argv, required, &provided));

    if (provided < required && config.debug) {
        std::fprintf(stderr, "MPI: Thread level %d not available, got %d\n",
                     required, provided);
    }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size_));

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_CHECK(MPI_Get_processor_name(name, &len));
    hostname_ = std::string(name, len);

    config_ = config;
    compute_local_rank();
    compute_node_id();
#else
    (void)argc;
    (void)argv;
    (void)config;
    world_rank_ = 0;
    world_size_ = 1;
    local_rank_ = 0;
    local_size_ = 1;
    node_id_ = 0;
#endif

    initialized_ = true;
}

void MpiContext::finalize() {
    std::lock_guard<std::mutex> lock(init_mutex);

    if (!initialized_) {
        return;
    }

#if NOVA_MPI_ENABLED
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_CHECK(MPI_Finalize());
    }
#endif

    initialized_ = false;
    world_rank_ = 0;
    world_size_ = 1;
    local_rank_ = 0;
    local_size_ = 1;
}

void MpiContext::compute_local_rank() {
#if NOVA_MPI_ENABLED
    gather_hostnames();

    std::vector<int> ranks(world_size_);
    std::vector<int> displs(world_size_);
    std::vector<char> all_names(MPI_MAX_PROCESSOR_NAME * world_size_);

    for (int i = 0; i < world_size_; ++i) {
        displs[i] = i * MPI_MAX_PROCESSOR_NAME;
    }

    MPI_CHECK(MPI_Allgatherv(hostname_.c_str(),
                              static_cast<int>(hostname_.size()),
                              MPI_CHAR,
                              all_names.data(),
                              nullptr,
                              displs.data(),
                              MPI_CHAR,
                              MPI_COMM_WORLD));

    std::string my_hostname_padded = hostname_;
    my_hostname_padded.resize(MPI_MAX_PROCESSOR_NAME, '\0');

    local_rank_ = 0;
    for (int i = 0; i < world_rank_; ++i) {
        std::string other(&all_names[i * MPI_MAX_PROCESSOR_NAME],
                          MPI_MAX_PROCESSOR_NAME);
        if (other == my_hostname_padded) {
            local_rank_++;
        }
    }
#endif
}

void MpiContext::gather_hostnames() {
#if NOVA_MPI_ENABLED
    char my_name[MPI_MAX_PROCESSOR_NAME];
    int my_len;
    MPI_Get_processor_name(my_name, &my_len);
    hostname_ = std::string(my_name, my_len);

    std::vector<int> recvcounts(world_size_, MPI_MAX_PROCESSOR_NAME);
    std::vector<int> displs(world_size_);
    for (int i = 0; i < world_size_; ++i) {
        displs[i] = i * MPI_MAX_PROCESSOR_NAME;
    }

    std::vector<char> all_names(MPI_MAX_PROCESSOR_NAME * world_size_);
    MPI_Allgatherv(my_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                   all_names.data(), recvcounts.data(), displs.data(),
                   MPI_CHAR, MPI_COMM_WORLD);

    std::set<std::string> unique_nodes;
    for (int i = 0; i < world_size_; ++i) {
        std::string name(&all_names[i * MPI_MAX_PROCESSOR_NAME], my_len);
        unique_nodes.insert(name);
        if (i == world_rank_) {
            node_id_ = static_cast<int>(unique_nodes.size()) - 1;
        }
    }
    local_size_ = static_cast<int>(unique_nodes.size());
#endif
}

void MpiContext::compute_node_id() {
#if NOVA_MPI_ENABLED
    gather_hostnames();
#else
    node_id_ = 0;
    local_size_ = 1;
#endif
}

int MpiContext::get_local_device_id() const {
    const char* cuda_visible = std::getenv("CUDA_VISIBLE_DEVICES");
    if (cuda_visible && strlen(cuda_visible) > 0) {
        return local_rank_;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    return local_rank_ % device_count;
}

void validate_mpi_environment() {
#if NOVA_MPI_ENABLED
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);

    if (finalized) {
        throw std::runtime_error("MPI already finalized before context creation");
    }

    if (!initialized) {
        throw std::runtime_error(
            "MPI not initialized. Call MpiContext::initialize() first");
    }
#endif
}

}  // namespace cuda::mpi
