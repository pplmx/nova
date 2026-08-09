#include "cuda/production/graph_executor.h"

#include <cstring>
#include <utility>

namespace cuda::production {

GraphExecutor::GraphExecutor() {
    CUDA_CHECK(cudaGraphCreate(&graph_, 0));
}

GraphExecutor::GraphExecutor(unsigned int flags) {
    CUDA_CHECK(cudaGraphCreate(&graph_, flags));
}

GraphExecutor::~GraphExecutor() {
    cleanup_exec();
    cleanup_graph();
}

GraphExecutor::GraphExecutor(GraphExecutor&& other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)),
      exec_(std::exchange(other.exec_, nullptr)),
      capture_owner_(std::move(other.capture_owner_)),
      capture_stream_(std::exchange(other.capture_stream_, nullptr)),
      capturing_(std::exchange(other.capturing_, false)),
      instantiated_(std::exchange(other.instantiated_, false)),
      params_(std::move(other.params_)),
      param_nodes_(std::move(other.param_nodes_)) {}

GraphExecutor& GraphExecutor::operator=(GraphExecutor&& other) noexcept {
    if (this != &other) {
        cleanup_exec();
        cleanup_graph();

        graph_ = std::exchange(other.graph_, nullptr);
        exec_ = std::exchange(other.exec_, nullptr);
        capture_owner_ = std::move(other.capture_owner_);
        capture_stream_ = std::exchange(other.capture_stream_, nullptr);
        capturing_ = std::exchange(other.capturing_, false);
        instantiated_ = std::exchange(other.instantiated_, false);
        params_ = std::move(other.params_);
        param_nodes_ = std::move(other.param_nodes_);
    }
    return *this;
}

void GraphExecutor::begin_capture(cudaStream_t stream) {
    validate_state();

    if (capturing_) {
        throw device::CudaException(cudaErrorInvalidValue, __FILE__, __LINE__);
    }

    capture_stream_ = stream;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    capturing_ = true;
}

void GraphExecutor::begin_capture(cuda::stream::Stream& stream) {
    begin_capture(stream.get());
}

void GraphExecutor::begin_capture() {
    // Own the capture stream inside the executor. A stack-local Stream here
    // would be destroyed when begin_capture() returns, leaving capture_stream_
    // dangling; end_capture() then passes a destroyed stream handle to
    // cuStreamEndCapture (use-after-free -> SIGSEGV inside libcuda).
    capture_owner_.emplace();
    begin_capture(*capture_owner_);
}

cudaGraph_t GraphExecutor::end_capture() {
    if (!capturing_) {
        throw device::CudaException(cudaErrorInvalidValue, __FILE__, __LINE__);
    }

    CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
    capturing_ = false;

    return graph_;
}

void GraphExecutor::instantiate() {
    if (instantiated_) {
        cleanup_exec();
    }

    if (graph_ == nullptr) {
        throw device::CudaException(cudaErrorInvalidValue, __FILE__, __LINE__);
    }

    CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
    instantiated_ = true;
}

void GraphExecutor::launch(cudaStream_t stream) {
    if (!instantiated_) {
        throw device::CudaException(cudaErrorInvalidValue, __FILE__, __LINE__);
    }

    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

void GraphExecutor::launch(cuda::stream::Stream& stream) {
    launch(stream.get());
}

void GraphExecutor::launch() {
    cuda::stream::Stream default_stream;
    launch(default_stream);
}

void GraphExecutor::update_param(size_t index, const void* param, size_t size) {
    if (index >= params_.size()) {
        params_.resize(index + 1);
    }

    auto& p = params_[index];
    p.resize(size);
    if (param != nullptr && size > 0) {
        std::memcpy(p.data(), param, size);
    }
}

void GraphExecutor::reset() {
    cleanup_exec();
    cleanup_graph();

    // Release any internally-owned capture stream and its dangling alias.
    capture_owner_.reset();
    capture_stream_ = nullptr;
    params_.clear();
    param_nodes_.clear();
    capturing_ = false;
    instantiated_ = false;

    CUDA_CHECK(cudaGraphCreate(&graph_, 0));
}

void GraphExecutor::validate_state() const {
    if (graph_ == nullptr) {
        throw device::CudaException(cudaErrorInvalidValue, __FILE__, __LINE__);
    }
}

void GraphExecutor::cleanup_graph() {
    if (graph_ != nullptr) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

void GraphExecutor::cleanup_exec() {
    if (exec_ != nullptr) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
}

ScopedCapture::ScopedCapture(GraphExecutor& executor, cudaStream_t stream)
    : executor_(executor) {
    executor_.begin_capture(stream);
}

ScopedCapture::ScopedCapture(GraphExecutor& executor) : executor_(executor) {
    executor_.begin_capture();
}

ScopedCapture::~ScopedCapture() {
    try {
        executor_.end_capture();
        executor_.instantiate();
    } catch (...) {
    }
}

}  // namespace cuda::production
