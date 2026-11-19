#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cuda/production/graph_executor.h"

namespace cuda::production {

    template <typename T>
    class GraphReduceWrapper {
    public:
        using ReduceFn = std::function<T(const T*, size_t, T)>;

        GraphReduceWrapper() = default;

        explicit GraphReduceWrapper(ReduceFn fn)
            : reduce_fn_(std::move(fn)) {}

        void set_reducer(ReduceFn fn) { reduce_fn_ = std::move(fn); }

        void capture(cuda::stream::Stream& stream);

        void launch(cuda::stream::Stream& stream);

        [[nodiscard]] T result() const { return result_; }

        ~GraphReduceWrapper();

    private:
        ReduceFn reduce_fn_;
        GraphExecutor executor_;
        T result_ {};
        size_t input_size_ = 0;
        // Owned here so it can be allocated before capture begins and freed after
        // the graph is gone; allocating inside a capturing stream is an error.
        T* reduce_scratch_ = nullptr;
    };

    template <typename T>
    class GraphScanWrapper {
    public:
        using ScanFn = std::function<void(T*, const T*, size_t, bool)>;

        GraphScanWrapper() = default;

        explicit GraphScanWrapper(ScanFn fn)
            : scan_fn_(std::move(fn)) {}

        void set_scanner(ScanFn fn) { scan_fn_ = std::move(fn); }

        void capture(cuda::stream::Stream& stream);

        void launch(cuda::stream::Stream& stream);

        ~GraphScanWrapper();

        /**
         * @brief Output buffer of the last capture
         *
         * Not implemented: GraphScanWrapper stores no buffer, so this always
         * returns nullptr. Making it return something meaningful requires the scan
         * to own its output allocation and to pass it to scan_fn_, which the current
         * design does not do. Documented rather than silently returning null.
         */
        [[nodiscard]] T* output() const { return nullptr; }

    private:
        ScanFn scan_fn_;
        GraphExecutor executor_;
        size_t input_size_ = 0;
    };

    template <typename T>
    class GraphSortWrapper {
    public:
        using SortFn = std::function<void(T*, size_t, bool)>;

        GraphSortWrapper() = default;

        explicit GraphSortWrapper(SortFn fn)
            : sort_fn_(std::move(fn)) {}

        void set_sorter(SortFn fn) { sort_fn_ = std::move(fn); }

        void capture(cuda::stream::Stream& stream);

        void launch(cuda::stream::Stream& stream);

        ~GraphSortWrapper();

        /**
         * @brief Data buffer of the last capture
         *
         * Not implemented: GraphSortWrapper stores no buffer, so this always
         * returns nullptr. See GraphScanWrapper::output() for the same limitation.
         */
        [[nodiscard]] T* data() const { return nullptr; }

    private:
        SortFn sort_fn_;
        GraphExecutor executor_;
        size_t data_size_ = 0;
    };

    template <typename T>
    class GraphAlgoContext {
    public:
        GraphAlgoContext() = default;

        /**
         * @brief Register a named algorithm for capture/launch
         *
         * Defined inline: as a template, its body has to be reachable from every
         * call site. With the definition in the .cu file every external call failed
         * to link ("undefined reference to wrap_algo<int>").
         *
         * @note `algo` is accepted for API compatibility but is not stored. The
         *       entry records only the name and its own GraphExecutor; the executor
         *       has no way to invoke the algorithm, which is why capture_all below
         *       does not produce a runnable graph.
         */
        template <typename Algo>
        void wrap_algo(const char* name, Algo&& algo) {
            (void)algo;
            AlgoEntry entry;
            entry.name = name != nullptr ? name : "";
            algos_.push_back(std::move(entry));
        }

        void capture_all(cuda::stream::Stream& stream);
        void launch_all(cuda::stream::Stream& stream);

        [[nodiscard]] size_t algo_count() const { return algos_.size(); }

    private:
        struct AlgoEntry {
            std::string name;
            GraphExecutor executor;
        };

        std::vector<AlgoEntry> algos_;
    };

    template <typename T>
    __global__ void identity_kernel(T* out, const T* in, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = in[idx];
        }
    }

}  // namespace cuda::production
