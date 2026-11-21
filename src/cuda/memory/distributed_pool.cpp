#include "cuda/memory/distributed_pool.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <cstring>
#include <sstream>

#include "cuda/device/error.h"
#include "cuda/observability/logger.hpp"

namespace cuda::memory {

    // Default constructor uses default config - pools initialized lazily on first allocation
    DistributedMemoryPool::DistributedMemoryPool() {
        initialize_pools();
    }

    // Explicit config constructor - allows tuning block_size and max_blocks_per_device
    // before any allocations occur
    DistributedMemoryPool::DistributedMemoryPool(const Config& config)
        : config_(config) {
        initialize_pools();
    }

    // Destructor ensures clean shutdown - all allocations released via clear()
    DistributedMemoryPool::~DistributedMemoryPool() {
        clear();
    }

    // Move constructor transfers pool state including device memory tracking
    // Note: ownership_map_ requires mutex protection even after move (handled by lock in move assignment)
    DistributedMemoryPool::DistributedMemoryPool(DistributedMemoryPool&& other) noexcept
        : config_(std::move(other.config_)),
          pools_(std::move(other.pools_)),
          ownership_map_(std::move(other.ownership_map_)) {
        // ownership_map_ moved but needs mutex - done via assignment
    }

    // Move assignment clears existing state first to release any held allocations
    // This prevents memory leaks when pool is reassigned
    DistributedMemoryPool& DistributedMemoryPool::operator=(DistributedMemoryPool&& other) noexcept {
        if (this != &other) {
            clear();
            config_ = std::move(other.config_);
            pools_ = std::move(other.pools_);
            ownership_map_ = std::move(other.ownership_map_);
        }
        return *this;
    }

    // Initialize per-device memory pools after detecting available hardware
    // This is called from constructors - must succeed before pool is usable
    void DistributedMemoryPool::initialize_pools() {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices available");
        }

        // Reserve capacity to avoid reallocations during pool creation
        pools_.reserve(device_count);

        // Create a memory pool for each device and query device memory capacity
        // Each pool is configured identically based on config_ settings
        for (int i = 0; i < device_count; ++i) {
            MemoryPool::Config pool_config;
            pool_config.block_size = config_.block_size;
            pool_config.max_blocks = config_.max_blocks_per_device;

            pools_.emplace_back(pool_config);
        }

        // Nothing about the devices' memory is cached: get_best_device() and
        // get_device_metrics() query the driver when asked, because free memory changes as
        // every tenant on the machine allocates and frees. Caching it at construction was
        // what made both report a number that was not free memory.
    }

    // Query the free and total memory of one device by switching context temporarily.
    //
    // The name says "free" while the body returns total_mem, and every caller then subtracted
    // this pool's OWN allocations from it. That made the result "total minus our own use",
    // which is only free memory on an idle machine: with another tenant holding 8 GB of
    // device 0, this reported 85.1 GB free where cudaMemGetInfo said 76.1 GB.
    //
    // The device switch is necessary because cudaMemGetInfo reports the current device only.
    // Both numbers are returned now, so a caller cannot silently get one for the other.
    DistributedMemoryPool::DeviceMemory DistributedMemoryPool::query_device_memory(int device_id) const {
        size_t free_mem = 0;
        size_t total_mem = 0;

        // Save current device so we can restore it after querying
        int current_device = 0;
        CUDA_CHECK(cudaGetDevice(&current_device));

        // Switch to target device to query memory
        // This is a blocking operation but necessary for accurate memory queries
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        // Restore device state for caller
        CUDA_CHECK(cudaSetDevice(current_device));

        return DeviceMemory {.free_bytes = free_mem, .total_bytes = total_mem};
    }

    // Allocate memory from a specific device's pool
    // Thread-safe via per-device pool locking and global ownership_map_ mutex
    void* DistributedMemoryPool::allocate(size_t bytes, int device_id, int stream_id) {
        if (device_id < 0 || device_id >= device_count()) {
            NOVA_LOG_ERROR("component=memory_pool", ("Invalid device ID: " + std::to_string(device_id)).c_str());
            throw std::invalid_argument("Invalid device ID: " + std::to_string(device_id));
        }

        // Allocate from the specified device's pool
        void* ptr = pools_[device_id].allocate(bytes, stream_id);

        // Track ownership to support cross-device deallocation
        // This map enables deallocate() to find the correct device pool
        OwnershipRecord record;
        record.ptr = ptr;
        record.bytes = bytes;
        record.owning_device = device_id;
        record.requesting_device = device_id;

        std::lock_guard<std::mutex> lock(ownership_mutex_);
        ownership_map_[ptr] = record;

        // Log successful allocation at INFO level
        NOVA_LOG_INFO("operation=allocate", ("bytes=" + std::to_string(bytes) + " device=" + std::to_string(device_id) + " stream=" + std::to_string(stream_id)).c_str());

        return ptr;
    }

    // Allocate on the device with most available memory when auto-allocation enabled
    // Fallback to device 0 when disabled ensures consistent behavior for non-multi-GPU code
    void* DistributedMemoryPool::allocate_auto(size_t bytes, int stream_id) {
        if (config_.enable_auto_allocation) {
            return allocate(bytes, get_best_device(), stream_id);
        }
        // Fallback to device 0
        return allocate(bytes, 0, stream_id);
    }

    // Find the device with the most free memory for load balancing
    //
    // The header promises "most free memory", and the old body computed
    // total - this pool's own allocations. That ignores every other tenant on the machine, so
    // it ranked devices by a number that is not free memory. Measured with 60 GB of device 0
    // and 30 GB of device 1 held by another allocation while device 2 was untouched:
    //
    //     dev 0: real_free=20.1 GB   dev 1: 52.4 GB   dev 2: 84.6 GB
    //     get_best_device() = 0      <- the device with the LEAST free memory
    //
    // Every assignment went to the busiest device, which is the opposite of the policy the
    // method exists to implement. It asks the driver now; this pool's own use is already
    // reflected in what cudaMemGetInfo reports, so subtracting it again would double-count.
    int DistributedMemoryPool::get_best_device() const {
        int best_device = 0;
        size_t max_free = 0;

        for (int i = 0; i < device_count(); ++i) {
            const size_t free_on_device = query_device_memory(i).free_bytes;
            if (free_on_device > max_free) {
                max_free = free_on_device;
                best_device = i;
            }
        }

        return best_device;
    }

    // Deallocate memory - must find owning device from ownership_map_
    // This enables callers to deallocate without tracking which device allocation came from
    void DistributedMemoryPool::deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(ownership_mutex_);

        auto it = ownership_map_.find(ptr);
        if (it == ownership_map_.end()) {
            throw std::invalid_argument("Attempt to deallocate unknown pointer");
        }

        const OwnershipRecord& record = it->second;

        // Deallocate from owning device's pool (not necessarily current device)
        pools_[record.owning_device].deallocate(ptr, record.bytes);

        // Remove from ownership map to prevent double-deallocation
        ownership_map_.erase(it);
    }

    // Get ownership record for a pointer - used for debugging and memory tracking
    // Returns default record with owning_device=-1 if pointer not found
    OwnershipRecord DistributedMemoryPool::get_ownership(void* ptr) const {
        std::lock_guard<std::mutex> lock(ownership_mutex_);

        auto it = ownership_map_.find(ptr);
        if (it == ownership_map_.end()) {
            return {nullptr, 0, -1, -1};
        }
        return it->second;
    }

    // Quick check if pointer belongs to any pool - faster than get_ownership()
    bool DistributedMemoryPool::owns_pointer(void* ptr) const {
        std::lock_guard<std::mutex> lock(ownership_mutex_);
        return ownership_map_.find(ptr) != ownership_map_.end();
    }

    // Get memory metrics for a specific device
    // Useful for monitoring and debugging memory usage patterns
    DevicePoolMetrics DistributedMemoryPool::get_device_metrics(int device_id) const {
        if (device_id < 0 || device_id >= device_count()) {
            throw std::invalid_argument("Invalid device ID");
        }

        DevicePoolMetrics metrics;
        metrics.device_id = device_id;
        metrics.allocated_bytes = pools_[device_id].total_allocated();

        // Both figures come from the driver, so they describe the machine rather than this
        // pool's share of it. The previous code reported total - own_allocations under the
        // name free_memory, which overstates what a caller can allocate by whatever every
        // other tenant on the device is holding.
        const DeviceMemory memory = query_device_memory(device_id);
        metrics.total_memory = memory.total_bytes;
        metrics.free_memory = memory.free_bytes;
        // Room this pool could still hand out before the device is full, given that the pool
        // cannot reclaim another owner's blocks.
        metrics.available_bytes = memory.free_bytes + pools_[device_id].total_allocated();
        metrics.num_allocations = static_cast<int>(pools_[device_id].num_allocations());
        metrics.local_metrics = pools_[device_id].get_metrics();

        return metrics;
    }

    // Get metrics for all devices in one call - more efficient than individual calls
    // when you need the full picture
    std::vector<DevicePoolMetrics> DistributedMemoryPool::get_all_metrics() const {
        std::vector<DevicePoolMetrics> all_metrics;
        all_metrics.reserve(device_count());

        for (int i = 0; i < device_count(); ++i) {
            all_metrics.push_back(get_device_metrics(i));
        }

        return all_metrics;
    }

    // Clear all pools and reset ownership tracking
    // Called from destructor and move assignment to ensure clean state
    void DistributedMemoryPool::clear() {
        std::lock_guard<std::mutex> lock(ownership_mutex_);

        // Clear ownership map first to prevent any race conditions during pool cleanup
        ownership_map_.clear();

        // Clear all pools - each pool frees its blocks
        for (auto& pool : pools_) {
            pool.clear();
        }
    }

}  // namespace cuda::memory
