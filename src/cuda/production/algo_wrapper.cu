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
    const cudaStream_t stream_handle = stream.get();
    // cudaMalloc/cudaFree are host-synchronising calls and are NOT permitted
    // while a stream is capturing: the runtime aborts with "operation not
    // permitted when stream is capturing". Allocate the scratch buffer BEFORE
    // begin_capture, and keep it alive until the wrapper is destroyed.
    if (reduce_scratch_ == nullptr) {
        CUDA_CHECK(cudaMalloc(&reduce_scratch_, sizeof(T)));
    }

    executor_.begin_capture(stream_handle);

    if (reduce_fn_) {
        reduce_fn_(nullptr, input_size_, T{});
    }

    set_value_kernel<<<1, 1, 0, stream_handle>>>(reduce_scratch_, T{}, 1);

    (void)executor_.end_capture();
    executor_.instantiate();
}

template <typename T>
GraphReduceWrapper<T>::~GraphReduceWrapper() {
    // cudaFree outside any capture is fine.
    if (reduce_scratch_ != nullptr) {
        cudaFree(reduce_scratch_);
        reduce_scratch_ = nullptr;
    }
}

template <typename T>
void GraphReduceWrapper<T>::launch(cuda::stream::Stream& stream) {
    executor_.launch(stream.get());
    stream.synchronize();
}

template <typename T>
GraphScanWrapper<T>::~GraphScanWrapper() = default;

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
GraphSortWrapper<T>::~GraphSortWrapper() = default;

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
void GraphAlgoContext<T>::capture_all(cuda::stream::Stream& stream) {
    const cudaStream_t stream_handle = stream.get();

    // Each entry opened its own capture on the same stream and none was ever
    // closed, so the stream stayed in capture mode and the very next CUDA call
    // failed with "operation not permitted when stream is capturing". Capture
    // and close one entry at a time.
    //
    // Note: an entry's executor has no recorded work (wrap_algo stores only a
    // name), so these graphs are empty. What this fixes is the leaked capture
    // state, which broke every subsequent call on the stream.
    for (auto& algo : algos_) {
        algo.executor.begin_capture(stream_handle);
        (void)algo.executor.end_capture();
        algo.executor.instantiate();
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
