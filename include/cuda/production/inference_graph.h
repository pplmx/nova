#pragma once

#include "cuda/production/graph_executor.h"
#include "cuda/inference/block_manager.h"
#include <memory>

namespace cuda::production {

class InferenceGraphExecutor : public GraphExecutor {
public:
    explicit InferenceGraphExecutor(
        const inference::BlockManager& block_manager,
        int max_batch_size = 32,
        int max_seq_len = 512
    );

    void capture_inference_pass(
        const std::vector<int64_t>& sequence_ids,
        const memory::Buffer<float>& query,
        memory::Buffer<float>& output,
        const stream::Stream& capture_stream
    );

    void replay_inference(
        const memory::Buffer<float>& query,
        memory::Buffer<float>& output
    );

    void update_batch_size(int new_batch_size);

    int get_current_batch_size() const { return current_batch_size_; }
    bool is_captured() const { return graph_captured_; }

private:
    const inference::BlockManager& block_manager_;
    int current_batch_size_ = 0;
    bool graph_captured_ = false;
};

InferenceGraphExecutor::InferenceGraphExecutor(
    const inference::BlockManager& block_manager,
    int max_batch_size,
    int max_seq_len
) : block_manager_(block_manager),
    current_batch_size_(max_batch_size) {

    GraphExecutorConfig config{
        .max_batch_size = max_batch_size,
        .enable_monitoring = true
    };

    init(config);
}

void InferenceGraphExecutor::capture_inference_pass(
    const std::vector<int64_t>& sequence_ids,
    const memory::Buffer<float>& query,
    memory::Buffer<float>& output,
    const stream::Stream& capture_stream
) {
    begin_capture(capture_stream);

    block_manager_.forward_batch(sequence_ids, query, output, capture_stream);

    CUDA_CHECK(cudaStreamSynchronize(capture_stream.get()));

    end_capture();

    graph_captured_ = true;
    current_batch_size_ = static_cast<int>(sequence_ids.size());
}

void InferenceGraphExecutor::replay_inference(
    const memory::Buffer<float>& query,
    memory::Buffer<float>& output
) {
    if (!graph_captured_) {
        throw std::runtime_error("Graph not captured - call capture_inference_pass first");
    }

    replay();
}

void InferenceGraphExecutor::update_batch_size(int new_batch_size) {
    if (new_batch_size != current_batch_size_) {
        current_batch_size_ = new_batch_size;
        graph_captured_ = false;
    }
}

}  // namespace cuda::production
