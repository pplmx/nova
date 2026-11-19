#pragma once

#include <cuda_runtime.h>

#include <optional>

#include "cuda/device/error.h"

namespace cuda::production {

    enum class CachePreference {
        NoPreference,
        PreferShared,
        PreferL1,
        Equal,
    };

    class L2PersistenceManager {
    public:
        L2PersistenceManager() = default;

        explicit L2PersistenceManager(size_t persistence_size);

        ~L2PersistenceManager() { restore_defaults(); }

        L2PersistenceManager(const L2PersistenceManager&) = delete;
        L2PersistenceManager& operator=(const L2PersistenceManager&) = delete;

        L2PersistenceManager(L2PersistenceManager&& other) noexcept
            : saved_config_(std::exchange(other.saved_config_, std::nullopt)),
              active_(std::exchange(other.active_, false)) {}

        L2PersistenceManager& operator=(L2PersistenceManager&& other) noexcept {
            if (this != &other) {
                restore_defaults();
                saved_config_ = std::exchange(other.saved_config_, std::nullopt);
                active_ = std::exchange(other.active_, false);
            }
            return *this;
        }

        void set_persistence_size(size_t bytes);
        void restore_defaults();

        [[nodiscard]] bool is_active() const { return active_; }
        [[nodiscard]] size_t persistence_size() const { return persistence_size_; }
        [[nodiscard]] size_t max_persistence_size() const;

    private:
        std::optional<cudaFuncCache> saved_config_;
        bool active_ = false;
        size_t persistence_size_ = 0;
    };

    class ScopedL2Persistence {
    public:
        explicit ScopedL2Persistence(size_t bytes);
        ~ScopedL2Persistence() { manager_.restore_defaults(); }

        ScopedL2Persistence(const ScopedL2Persistence&) = delete;
        ScopedL2Persistence& operator=(const ScopedL2Persistence&) = delete;

        ScopedL2Persistence(ScopedL2Persistence&&) = default;
        ScopedL2Persistence& operator=(ScopedL2Persistence&&) = default;

    private:
        L2PersistenceManager manager_;
    };

}  // namespace cuda::production
