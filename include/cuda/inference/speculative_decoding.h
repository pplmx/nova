#pragma once

#include "cuda/inference/block_manager.h"
#include "cuda/inference/beam_search.h"
#include "cuda/memory/buffer.h"
#include "cuda/stream/stream.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace cuda::inference {

struct SpeculativeDecodingConfig {
    int draft_depth = 4;
    float acceptance_threshold = 0.8f;
    bool enable_tree_attention = true;
    bool enable_async_draft = true;
    bool enable_eagle3 = false;
    bool enable_xgrammar = false;
    int max_draft_depth = 8;
};

struct DraftToken {
    int token_id;
    float draft_prob;
    float target_prob;
    bool accepted;
};

struct VerificationResult {
    std::vector<DraftToken> tokens;
    int num_accepted;
    float kl_divergence;
};

struct LogProbEntry {
    int token_id;
    float log_prob;
    float draft_log_prob;
    bool accepted;
};

class LogProbTracker {
public:
    void record(
        int token_id,
        float log_prob,
        float draft_log_prob,
        bool accepted
    );

    float compute_total_kl_divergence() const;

    float compute_average_kl_divergence() const;

    int num_accepted() const;

    int num_rejected() const;

    void clear();

    std::vector<LogProbEntry> get_history() const { return history_; }

private:
    std::vector<LogProbEntry> history_;
};

class SpeculativeDecodingRunner {
public:
    explicit SpeculativeDecodingRunner(
        BlockManager* block_manager,
        const SpeculativeDecodingConfig& config
    );

    SpeculativeDecodingRunner(const SpeculativeDecodingRunner&) = delete;
    SpeculativeDecodingRunner& operator=(const SpeculativeDecodingRunner&) = delete;
    SpeculativeDecodingRunner(SpeculativeDecodingRunner&&) = default;
    SpeculativeDecodingRunner& operator=(SpeculativeDecodingRunner&&) = default;

    std::vector<int> decode(
        const memory::Buffer<float>& prompt_embeddings,
        int prompt_length,
        const stream::Stream& stream,
        std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, bool, const stream::Stream&)>
            forward_fn
    );

    void configure(const SpeculativeDecodingConfig& config);
    SpeculativeDecodingConfig get_config() const { return config_; }

    VerificationResult verify_draft_tokens(
        const std::vector<int>& draft_tokens,
        const memory::Buffer<float>& draft_logits,
        const memory::Buffer<float>& target_logits,
        const stream::Stream& stream
    );

    LogProbTracker& get_logprob_tracker() { return logprob_tracker_; }

    void snapshot_kv_state();
    void rollback_kv_state();
    void commit_kv_state();

    float compute_kl_divergence(
        const std::vector<DraftToken>& tokens,
        const memory::Buffer<float>& draft_logits,
        const memory::Buffer<float>& target_logits
    ) const;

private:
    std::vector<int> generate_draft_tokens(
        const memory::Buffer<float>& prompt_embeddings,
        int prompt_length,
        const stream::Stream& stream,
        std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, bool, const stream::Stream&)>
            forward_fn
    );

    void apply_tree_attention_mask(
        int num_draft_tokens,
        const stream::Stream& stream
    );

    BlockManager* block_manager_;
    SpeculativeDecodingConfig config_;

    std::vector<int> draft_tokens_;
    std::vector<DraftToken> verification_results_;
    std::vector<LogProbEntry> logprob_history_;

    LogProbTracker logprob_tracker_;

    struct KVCacheSnapshot {
        std::vector<int64_t> sequence_ids;
        std::vector<int> num_blocks;
    };
    std::optional<KVCacheSnapshot> kv_snapshot_;
};

}  // namespace cuda::inference
