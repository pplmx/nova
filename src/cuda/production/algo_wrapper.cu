#include "cuda/production/algo_wrapper.h"

#include "cuda/device/error.h"

namespace cuda::production {

template <typename T>
__global__ void set_value_kernel(T* out, T value, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = value;
    }
}

template <typename T>
void GraphReduceWrapper<T>::capture(cuda::stream::Stream& stream) {
    auto stream_handle = stream.get();
    executor_.begin_capture(stream_handle);

    if (reduce_fn_) {
        reduce_fn_(nullptr, input_size_, T{});
    }

    T* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T)));
    set_value_kernel<<<1, 1, 0, stream_handle>>>(d_out, T{}, 1);

    (void)executor_.end_capture();
    executor_.instantiate();

    CUDA_CHECK(cudaFree(d_out));
}

template <typename T>
void GraphReduceWrapper<T>::launch(cuda::stream::Stream& stream) {
    executor_.launch(stream.get());
    stream.synchronize();
}

template <typename T>
void GraphScanWrapper<T>::capture(cuda::stream::Stream& stream) {
    executor_.begin_capture(stream.get());

    if (scan_fn_) {
        scan_fn_(nullptr, nullptr, input_size_, false);
    }

    (void)executor_.end_capture();
    executor_.instantiate();
}

template <typename T>
void GraphScanWrapper<T>::launch(cuda::stream::Stream& stream) {
    executor_.launch(stream.get());
    stream.synchronize();
}

template <typename T>
void GraphSortWrapper<T>::capture(cuda::stream::Stream& stream) {
    executor_.begin_capture(stream.get());

    if (sort_fn_) {
        sort_fn_(nullptr, data_size_, false);
    }

    (void)executor_.end_capture();
    executor_.instantiate();
}

template <typename T>
void GraphSortWrapper<T>::launch(cuda::stream::Stream& stream) {
    executor_.launch(stream.get());
    stream.synchronize();
}

template <typename T>
template <typename Algo>
void GraphAlgoContext<T>::wrap_algo(const char* name, Algo&& algo) {
    AlgoEntry entry;
    entry.name = name;
    algos_.push_back(std::move(entry));
}

template <typename T>
void GraphAlgoContext<T>::capture_all(cuda::stream::Stream& stream) {
    auto stream_handle = stream.get();
    for (auto& algo : algos_) {
        algo.executor.begin_capture(stream_handle);
    }
}

template <typename T>
void GraphAlgoContext<T>::launch_all(cuda::stream::Stream& stream) {
    auto stream_handle = stream.get();
    for (auto& algo : algos_) {
        algo.executor.launch(stream_handle);
    }
    stream.synchronize();
}

template class GraphReduceWrapper<float>;
template class GraphReduceWrapper<double>;
template class GraphReduceWrapper<int>;
template class GraphReduceWrapper<unsigned int>;

template class GraphScanWrapper<float>;
template class GraphScanWrapper<double>;
template class GraphScanWrapper<int>;
template class GraphScanWrapper<unsigned int>;

template class GraphSortWrapper<float>;
template class GraphSortWrapper<double>;
template class GraphSortWrapper<int>;
template class GraphSortWrapper<unsigned int>;

template class GraphAlgoContext<float>;
template class GraphAlgoContext<double>;
template class GraphAlgoContext<int>;
template class GraphAlgoContext<unsigned int>;

}  // namespace cuda::production
