#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <optional>
#include <vector>

#include "cuda/device/error.h"
#include "cuda/production/graph_executor.h"

namespace cuda::production {

    enum class MemoryType {
        Device,
        HostPinned,
        Managed,
    };

    class MemoryNode {
    public:
        MemoryNode() = default;
        MemoryNode(cudaGraphNode_t node, MemoryType type, void* ptr, size_t size)
            : node_(node),
              type_(type),
              ptr_(ptr),
              size_(size) {}

        [[nodiscard]] cudaGraphNode_t get() const { return node_; }
        [[nodiscard]] MemoryType type() const { return type_; }
        [[nodiscard]] void* ptr() const { return ptr_; }
        [[nodiscard]] size_t size() const { return size_; }

    private:
        cudaGraphNode_t node_{nullptr};
        MemoryType type_{MemoryType::Device};
        void* ptr_{nullptr};
        size_t size_{0};
    };

    class GraphMemoryManager {
    public:
        GraphMemoryManager() = default;

        [[nodiscard]] MemoryNode add_device_allocation(GraphExecutor& graph, cudaGraph_t cuda_graph, size_t size);

        [[nodiscard]] MemoryNode add_host_allocation(GraphExecutor& graph, cudaGraph_t cuda_graph, size_t size);

        [[nodiscard]] MemoryNode add_managed_allocation(GraphExecutor& graph, cudaGraph_t cuda_graph, size_t size);

        void free_device(void* ptr);
        void free_host(void* ptr);
        void free_managed(void* ptr);

        void cleanup();

        [[nodiscard]] size_t total_allocated() const { return total_allocated_; }
        [[nodiscard]] size_t allocation_count() const { return allocations_.size(); }

    private:
        struct Allocation {
            void* ptr;
            size_t size;
            MemoryType type;
        };

        std::vector<Allocation> allocations_;
        size_t total_allocated_ = 0;
    };

    template <typename T>
    class ScopedGraphBuffer {
    public:
        ScopedGraphBuffer() = default;

        ScopedGraphBuffer(GraphMemoryManager& manager, GraphExecutor& executor, cudaGraph_t graph, size_t count)
            : manager_(&manager),
              executor_(&executor),
              graph_(graph),
              count_(count) {
            auto node = manager_->add_device_allocation(*executor_, graph_, count * sizeof(T));
            ptr_ = static_cast<T*>(node.ptr());
        }

        ~ScopedGraphBuffer() {
            if (ptr_ != nullptr && manager_ != nullptr) {
                manager_->free_device(ptr_);
            }
        }

        ScopedGraphBuffer(const ScopedGraphBuffer&) = delete;
        ScopedGraphBuffer& operator=(const ScopedGraphBuffer&) = delete;

        ScopedGraphBuffer(ScopedGraphBuffer&& other) noexcept
            : manager_(other.manager_),
              executor_(other.executor_),
              graph_(other.graph_),
              count_(other.count_),
              ptr_(other.ptr_) {
            other.manager_ = nullptr;
            other.executor_ = nullptr;
            other.graph_ = nullptr;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        ScopedGraphBuffer& operator=(ScopedGraphBuffer&& other) noexcept {
            if (this != &other) {
                if (ptr_ != nullptr && manager_ != nullptr) {
                    manager_->free_device(ptr_);
                }
                manager_ = other.manager_;
                executor_ = other.executor_;
                graph_ = other.graph_;
                count_ = other.count_;
                ptr_ = other.ptr_;

                other.manager_ = nullptr;
                other.executor_ = nullptr;
                other.graph_ = nullptr;
                other.ptr_ = nullptr;
                other.count_ = 0;
            }
            return *this;
        }

        [[nodiscard]] T* get() const { return ptr_; }
        [[nodiscard]] T* data() const { return ptr_; }
        [[nodiscard]] size_t size() const { return count_; }

        [[nodiscard]] T& operator[](size_t i) { return ptr_[i]; }
        [[nodiscard]] const T& operator[](size_t i) const { return ptr_[i]; }

    private:
        GraphMemoryManager* manager_ = nullptr;
        GraphExecutor* executor_ = nullptr;
        cudaGraph_t graph_ = nullptr;
        size_t count_ = 0;
        T* ptr_ = nullptr;
    };

}  // namespace cuda::production
