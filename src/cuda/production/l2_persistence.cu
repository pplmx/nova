#include "cuda/production/l2_persistence.h"

namespace cuda::production {

    L2PersistenceManager::L2PersistenceManager(size_t persistence_size) {
        set_persistence_size(persistence_size);
    }

    void L2PersistenceManager::set_persistence_size(size_t bytes) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        size_t l2_size = prop.persistingL2CacheMaxSize;
        size_t max_bytes = l2_size;

        size_t config_size = bytes;
        if (config_size > max_bytes) {
            config_size = max_bytes;
        }

        int percent = static_cast<int>((config_size * 100) / max_bytes);
        if (percent > 100) {
            percent = 100;
        }

        cudaFuncCache cache_config;
        if (percent >= 48) {
            cache_config = cudaFuncCachePreferShared;
        } else if (percent >= 32) {
            cache_config = cudaFuncCachePreferL1;
        } else {
            cache_config = cudaFuncCachePreferNone;
        }

        CUDA_CHECK(cudaDeviceSetCacheConfig(cache_config));

        if (!saved_config_.has_value()) {
            cudaFuncCache current;
            CUDA_CHECK(cudaDeviceGetCacheConfig(&current));
            saved_config_ = current;
        }

        persistence_size_ = config_size;
        active_ = true;
    }

    void L2PersistenceManager::restore_defaults() {
        if (saved_config_.has_value()) {
            CUDA_CHECK(cudaDeviceSetCacheConfig(saved_config_.value()));
            saved_config_ = std::nullopt;
        }
        active_ = false;
        persistence_size_ = 0;
    }

    size_t L2PersistenceManager::max_persistence_size() const {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        return prop.persistingL2CacheMaxSize;
    }

    ScopedL2Persistence::ScopedL2Persistence(size_t bytes)
        : manager_(bytes) {}

}  // namespace cuda::production
