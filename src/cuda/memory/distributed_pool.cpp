#include "cuda/memory/distributed_pool.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <cstring>

#include "cuda/device/error.h"

namespace cuda::memory {

DistributedMemoryPool::DistributedMemoryPool() {
    initialize_pools();
}

DistributedMemoryPool::DistributedMemoryPool(const Config& config) : config_(config) {
    initialize_pools();
}

DistributedMemoryPool::~DistributedMemoryPool() {
    clear();
}

DistributedMemoryPool::DistributedMemoryPool(DistributedMemoryPool&& other) noexcept
    : config_(std::move(other.config_)),
      pools_(std::move(other.pools_)),
      device_total_memory_(std::move(other.device_total_memory_)),
      ownership_map_(std::move(other.ownership_map_)) {
    // ownership_map_ moved but needs mutex - done via assignment
}

DistributedMemoryPool& DistributedMemoryPool::operator=(DistributedMemoryPool&& other) noexcept {
    if (this != &other) {
        clear();
        config_ = std::move(other.config_);
        pools_ = std::move(other.pools_);
        device_total_memory_ = std::move(other.device_total_memory_);
        ownership_map_ = std::move(other.ownership_map_);
    }
    return *this;
}

void DistributedMemoryPool::initialize_pools() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    pools_.reserve(device_count);
    device_total_memory_.reserve(device_count);

    for (int i = 0; i < device_count; ++i) {
        MemoryPool::Config pool_config;
        pool_config.block_size = config_.block_size;
        pool_config.max_blocks = config_.max_blocks_per_device;

        pools_.emplace_back(pool_config);

        // Query total memory for this device
        size_t total_mem = query_device_free_memory(i);
        device_total_memory_.push_back(total_mem);
    }
}

size_t DistributedMemoryPool::query_device_free_memory(int device_id) const {
    size_t free_mem = 0;
    size_t total_mem = 0;

    // Save current device
    int current_device = 0;
    cudaGetDevice(&current_device);

    // Switch to target device to query memory
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);

    // Restore device
    cudaSetDevice(current_device);

    return total_mem;
}

void* DistributedMemoryPool::allocate(size_t bytes, int device_id, int stream_id) {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID: " + std::to_string(device_id));
    }

    // Allocate from the specified device
    void* ptr = pools_[device_id].allocate(bytes, stream_id);

    // Track ownership
    OwnershipRecord record;
    record.ptr = ptr;
    record.bytes = bytes;
    record.owning_device = device_id;
    record.requesting_device = device_id;

    std::lock_guard<std::mutex> lock(ownership_mutex_);
    ownership_map_[ptr] = record;

    return ptr;
}

void* DistributedMemoryPool::allocate_auto(size_t bytes, int stream_id) {
    if (config_.enable_auto_allocation) {
        return allocate(bytes, get_best_device(), stream_id);
    }
    // Fallback to device 0
    return allocate(bytes, 0, stream_id);
}

int DistributedMemoryPool::get_best_device() const {
    int best_device = 0;
    size_t max_free = 0;

    for (int i = 0; i < device_count(); ++i) {
        // Calculate free memory: total - allocated
        size_t allocated = pools_[i].total_allocated();
        size_t free_on_device = device_total_memory_[i] - allocated;

        if (free_on_device > max_free) {
            max_free = free_on_device;
            best_device = i;
        }
    }

    return best_device;
}

void DistributedMemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(ownership_mutex_);

    auto it = ownership_map_.find(ptr);
    if (it == ownership_map_.end()) {
        throw std::invalid_argument("Attempt to deallocate unknown pointer");
    }

    const OwnershipRecord& record = it->second;

    // Deallocate from owning device's pool
    pools_[record.owning_device].deallocate(ptr, record.bytes);

    // Remove from ownership map
    ownership_map_.erase(it);
}

OwnershipRecord DistributedMemoryPool::get_ownership(void* ptr) const {
    std::lock_guard<std::mutex> lock(ownership_mutex_);

    auto it = ownership_map_.find(ptr);
    if (it == ownership_map_.end()) {
        return {nullptr, 0, -1, -1};
    }
    return it->second;
}

bool DistributedMemoryPool::owns_pointer(void* ptr) const {
    std::lock_guard<std::mutex> lock(ownership_mutex_);
    return ownership_map_.find(ptr) != ownership_map_.end();
}

DevicePoolMetrics DistributedMemoryPool::get_device_metrics(int device_id) const {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }

    DevicePoolMetrics metrics;
    metrics.device_id = device_id;
    metrics.allocated_bytes = pools_[device_id].total_allocated();
    metrics.total_memory = device_total_memory_[device_id];
    metrics.available_bytes = device_total_memory_[device_id] - metrics.allocated_bytes;
    metrics.free_memory = metrics.available_bytes;
    metrics.num_allocations = static_cast<int>(pools_[device_id].num_allocations());
    metrics.local_metrics = pools_[device_id].get_metrics();

    return metrics;
}

std::vector<DevicePoolMetrics> DistributedMemoryPool::get_all_metrics() const {
    std::vector<DevicePoolMetrics> all_metrics;
    all_metrics.reserve(device_count());

    for (int i = 0; i < device_count(); ++i) {
        all_metrics.push_back(get_device_metrics(i));
    }

    return all_metrics;
}

void DistributedMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(ownership_mutex_);

    // Clear ownership map first
    ownership_map_.clear();

    // Clear all pools
    for (auto& pool : pools_) {
        pool.clear();
    }
}

}  // namespace cuda::memory
