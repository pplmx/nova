#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda/device/error.h"
#include "cuda/stream/stream.h"

namespace cuda::production {

class GraphNode {
public:
    GraphNode() = default;
    GraphNode(cudaGraphNode_t node, cudaGraphNodeType type)
        : node_(node), type_(type) {}

    [[nodiscard]] cudaGraphNode_t get() const { return node_; }
    [[nodiscard]] cudaGraphNodeType type() const { return type_; }

private:
    cudaGraphNode_t node_{nullptr};
    cudaGraphNodeType type_{cudaGraphNodeTypeKernel};
};

class ParameterNode {
public:
    ParameterNode() = default;
    explicit ParameterNode(cudaGraphNode_t node) : node_(node) {}

    [[nodiscard]] cudaGraphNode_t get() const { return node_; }

private:
    cudaGraphNode_t node_{nullptr};
};

class GraphExecutor {
public:
    static constexpr size_t MAX_PARAMS = 64;

    explicit GraphExecutor();
    explicit GraphExecutor(unsigned int flags);
    ~GraphExecutor();

    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;

    GraphExecutor(GraphExecutor&& other) noexcept;
    GraphExecutor& operator=(GraphExecutor&& other) noexcept;

    void begin_capture(cudaStream_t stream);
    void begin_capture(cuda::stream::Stream& stream);
    void begin_capture();

    [[nodiscard]] cudaGraph_t end_capture();

    void instantiate();

    void launch(cudaStream_t stream);
    void launch(cuda::stream::Stream& stream);
    void launch();

    void update_param(size_t index, const void* param, size_t size);

    void reset();

    [[nodiscard]] bool is_capturing() const { return capturing_; }
    [[nodiscard]] bool is_instantiated() const { return instantiated_; }
    [[nodiscard]] size_t param_count() const { return params_.size(); }

private:
    void validate_state() const;
    void cleanup_graph();
    void cleanup_exec();

    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t exec_{nullptr};

    // Owns the stream created by the no-arg begin_capture() so capture_stream_
    // never dangles. Captures begun with an explicit caller-provided stream do
    // not populate this.
    std::optional<cuda::stream::Stream> capture_owner_;
    cudaStream_t capture_stream_{nullptr};
    bool capturing_ = false;
    bool instantiated_ = false;

    std::vector<std::vector<unsigned char>> params_;
    std::vector<cudaGraphNode_t> param_nodes_;
};

class ScopedCapture {
public:
    explicit ScopedCapture(GraphExecutor& executor, cudaStream_t stream);
    explicit ScopedCapture(GraphExecutor& executor);
    ~ScopedCapture();

    ScopedCapture(const ScopedCapture&) = delete;
    ScopedCapture& operator=(const ScopedCapture&) = delete;

private:
    GraphExecutor& executor_;
};

}  // namespace cuda::production
