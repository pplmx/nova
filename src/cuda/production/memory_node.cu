#include "cuda/production/memory_node.h"

namespace cuda::production {

    MemoryNode GraphMemoryManager::add_device_allocation(cudaGraph_t cuda_graph, size_t size) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));

        // Anchor the allocation to the graph with a valid no-op node. The prior
        // implementation added a degenerate 3D *self*-copy memcpy node whose
        // cudaMemcpy3DParms lacked `extent`, so cudaGraphAddMemcpyNode always
        // failed with cudaErrorInvalidValue (the tests never reached this line:
        // every begin_capture()/end_capture() pair previously crashed on a
        // dangling capture stream). Graph memcpy nodes also cannot source host
        // memory, so no-op nodes are the correct placeholder for tracked
        // allocations.
        // The node is added and the allocation recorded under a guard: both can throw, and until the
        // push_back lands nothing but this frame knows about the memory cudaMalloc just handed out.
        // cudaGraphAddEmptyNode failing on a graph that is mid-capture or already instantiated is the
        // ordinary way to get here, and push_back itself can throw bad_alloc.
        cudaGraphNode_t node {};
        try {
            CUDA_CHECK(cudaGraphAddEmptyNode(&node, cuda_graph, nullptr, 0));
            allocations_.push_back({ptr, size, MemoryType::Device});
        } catch (...) {
            (void)cudaFree(ptr);
            ++released_count_;
            throw;
        }
        total_allocated_ += size;

        return MemoryNode(node, MemoryType::Device, ptr, size);
    }

    MemoryNode GraphMemoryManager::add_host_allocation(cudaGraph_t cuda_graph, size_t size) {
        void* host_ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_ptr, size));

        void* device_ptr = nullptr;
        try {
            CUDA_CHECK(cudaMalloc(&device_ptr, size));
        } catch (...) {
            (void)cudaFreeHost(host_ptr);
            ++released_count_;
            throw;
        }

        // Three things can fail after the two allocations land: the graph node, and either
        // push_back. All three leaked both buffers before this guard.
        cudaGraphNode_t node {};
        try {
            CUDA_CHECK(cudaGraphAddEmptyNode(&node, cuda_graph, nullptr, 0));
            allocations_.push_back({host_ptr, size, MemoryType::HostPinned});
            allocations_.push_back({device_ptr, size, MemoryType::Device});
        } catch (...) {
            (void)cudaFreeHost(host_ptr);
            (void)cudaFree(device_ptr);
            released_count_ += 2;
            throw;
        }
        total_allocated_ += size * 2;

        return MemoryNode(node, MemoryType::HostPinned, host_ptr, size);
    }

    MemoryNode GraphMemoryManager::add_managed_allocation(cudaGraph_t cuda_graph, size_t size) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocManaged(&ptr, size));

        cudaGraphNode_t node {};
        try {
            CUDA_CHECK(cudaGraphAddEmptyNode(&node, cuda_graph, nullptr, 0));
            allocations_.push_back({ptr, size, MemoryType::Managed});
        } catch (...) {
            (void)cudaFree(ptr);
            ++released_count_;
            throw;
        }
        total_allocated_ += size;

        return MemoryNode(node, MemoryType::Managed, ptr, size);
    }

    void GraphMemoryManager::free_device(void* ptr) {
        CUDA_CHECK(cudaFree(ptr));
        allocations_.erase(std::remove_if(allocations_.begin(), allocations_.end(), [ptr](const Allocation& a) { return a.ptr == ptr; }), allocations_.end());
    }

    void GraphMemoryManager::free_host(void* ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
        allocations_.erase(std::remove_if(allocations_.begin(), allocations_.end(), [ptr](const Allocation& a) { return a.ptr == ptr; }), allocations_.end());
    }

    void GraphMemoryManager::free_managed(void* ptr) {
        CUDA_CHECK(cudaFree(ptr));
        allocations_.erase(std::remove_if(allocations_.begin(), allocations_.end(), [ptr](const Allocation& a) { return a.ptr == ptr; }), allocations_.end());
    }

    void GraphMemoryManager::cleanup() {
        for (const auto& alloc : allocations_) {
            switch (alloc.type) {
                case MemoryType::Device:
                    cudaFree(alloc.ptr);
                    break;
                case MemoryType::HostPinned:
                    cudaFreeHost(alloc.ptr);
                    break;
                case MemoryType::Managed:
                    cudaFree(alloc.ptr);
                    break;
            }
        }
        allocations_.clear();
        total_allocated_ = 0;
    }

}  // namespace cuda::production
