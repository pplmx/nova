#pragma once

#include "cuda/inference/block_manager.h"
#include "cuda/memory/buffer.h"
#include "cuda/stream/stream.h"
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace cuda::inference {

struct BeamSearchConfig {
    int max_beams = 4;
    int min_beams = 1;
    float length_penalty = 0.7f;
    float temperature = 1.0f;
    int top_k = 1;
    float top_p = 1.0f;
    int max_length = 2048;
    bool enable_length_norm = true;
    bool enable_reuse = true;
};

struct BeamHypothesis {
    int64_t sequence_id;
    std::vector<int> tokens;
    float score;
    float log_prob;
    int length;
    bool finished;
    int parent_beam;
    int64_t kv_source_sequence;
};

struct BeamSearchTraceEntry {
    int step;
    int beam_id;
    int token;
    float score;
    float normalized_score;
    float log_prob;
    int64_t sequence_id;
};

class TopKSampler {
public:
    explicit TopKSampler(int k);

    std::vector<std::pair<int, float>> sample(
        const float* logits,
        int vocab_size,
        float temperature,
        uint64_t seed
    );

private:
    int k_;
};

class TopPSampler {
public:
    explicit TopPSampler(float p);

    int sample(
        const float* logits,
        int vocab_size,
        float temperature,
        uint64_t seed
    );

private:
    float p_;
};

class BeamSearchManager {
public:
    explicit BeamSearchManager(
        BlockManager* block_manager,
        const BeamSearchConfig& config
    );

    BeamSearchManager(const BeamSearchManager&) = delete;
    BeamSearchManager& operator=(const BeamSearchManager&) = delete;
    BeamSearchManager(BeamSearchManager&&) = default;
    BeamSearchManager& operator=(BeamSearchManager&&) = default;

    std::vector<BeamHypothesis> search(
        const memory::Buffer<float>& prompt_embeddings,
        int prompt_length,
        const stream::Stream& stream,
        std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, const stream::Stream&)>
            forward_fn
    );

    void configure(const BeamSearchConfig& config);
    BeamSearchConfig get_config() const { return config_; }

    std::string export_trace_json() const;
    std::string export_trace_csv() const;
    void clear_trace();

    struct TraceStats {
        int total_steps;
        int avg_beam_width;
        float avg_score;
        float avg_length_norm;
    };
    TraceStats get_trace_stats() const;

private:
    void initialize_beams(int prompt_length);
    void expand_beams(
        const memory::Buffer<float>& logits,
        const stream::Stream& stream
    );
    void prune_beams();
    void compute_length_normalized_scores();
    int sample_token(const float* logits, int vocab_size, uint64_t seed);

    BlockManager* block_manager_;
    BeamSearchConfig config_;
    std::vector<BeamHypothesis> hypotheses_;
    std::vector<BeamHypothesis> next_hypotheses_;

    std::vector<BeamSearchTraceEntry> trace_;
    int next_sequence_id_ = 1;

    std::mt19937 rng_;
};

}  // namespace cuda::inference
