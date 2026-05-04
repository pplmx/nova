#include "cuda/inference/speculative_decoding.h"
#include "cuda/device/error.h"
#include <algorithm>
#include <cmath>

namespace cuda::inference {

void LogProbTracker::record(
    int token_id,
    float log_prob,
    float draft_log_prob,
    bool accepted
) {
    LogProbEntry entry{
        .token_id = token_id,
        .log_prob = log_prob,
        .draft_log_prob = draft_log_prob,
        .accepted = accepted
    };
    history_.push_back(entry);
}

float LogProbTracker::compute_total_kl_divergence() const {
    float total_kl = 0.0f;
    for (const auto& entry : history_) {
        float p = std::exp(entry.log_prob);
        float q = std::exp(entry.draft_log_prob);
        if (p > 0.0f && q > 0.0f) {
            total_kl += p * std::log(p / q);
        }
    }
    return total_kl;
}

float LogProbTracker::compute_average_kl_divergence() const {
    if (history_.empty()) return 0.0f;
    return compute_total_kl_divergence() / history_.size();
}

int LogProbTracker::num_accepted() const {
    int count = 0;
    for (const auto& entry : history_) {
        if (entry.accepted) count++;
    }
    return count;
}

int LogProbTracker::num_rejected() const {
    return static_cast<int>(history_.size()) - num_accepted();
}

void LogProbTracker::clear() {
    history_.clear();
}

SpeculativeDecodingRunner::SpeculativeDecodingRunner(
    BlockManager* block_manager,
    const SpeculativeDecodingConfig& config
) : block_manager_(block_manager), config_(config), logprob_tracker_() {}

void SpeculativeDecodingRunner::configure(const SpeculativeDecodingConfig& config) {
    config_ = config;
}

void SpeculativeDecodingRunner::snapshot_kv_state() {
    KVCacheSnapshot snapshot;

    auto* kv_cache = block_manager_->get_kv_cache();
    const int num_free = kv_cache->get_num_free_blocks();

    kv_snapshot_ = KVCacheSnapshot{};
    kv_snapshot_->sequence_ids = {};
    kv_snapshot_->num_blocks = {};
}

void SpeculativeDecodingRunner::rollback_kv_state() {
    if (!kv_snapshot_) return;

    for (size_t i = 0; i < kv_snapshot_->sequence_ids.size(); ++i) {
        block_manager_->free_sequence(kv_snapshot_->sequence_ids[i]);
    }

    kv_snapshot_.reset();
}

void SpeculativeDecodingRunner::commit_kv_state() {
    kv_snapshot_.reset();
}

std::vector<int> SpeculativeDecodingRunner::generate_draft_tokens(
    const memory::Buffer<float>& prompt_embeddings,
    int prompt_length,
    const stream::Stream& stream,
    std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, bool, const stream::Stream&)>
        forward_fn
) {
    std::vector<int> draft_tokens;
    draft_tokens.reserve(config_.draft_depth);

    auto* seq = block_manager_->create_sequence(0, prompt_length + config_.max_draft_depth);
    int64_t seq_id = seq->id;

    memory::Buffer<float> dummy_output(512);

    for (int step = 0; step < config_.draft_depth; ++step) {
        std::vector<int64_t> seq_ids = {seq_id};
        forward_fn(dummy_output, seq_ids, false, stream);

        int token = 0;
        draft_tokens.push_back(token);

        block_manager_->append_tokens(seq_id, 1);
    }

    block_manager_->free_sequence(seq_id);

    return draft_tokens;
}

VerificationResult SpeculativeDecodingRunner::verify_draft_tokens(
    const std::vector<int>& draft_tokens,
    const memory::Buffer<float>& draft_logits,
    const memory::Buffer<float>& target_logits,
    const stream::Stream& stream
) {
    (void)stream;
    apply_tree_attention_mask(static_cast<int>(draft_tokens.size()), stream);

    VerificationResult result;
    result.tokens.reserve(draft_tokens.size());
    result.num_accepted = 0;
    result.kl_divergence = 0.0f;

    const float* draft_data = static_cast<const float*>(draft_logits.data());
    const float* target_data = static_cast<const float*>(target_logits.data());

    std::vector<float> draft_probs(draft_tokens.size());
    std::vector<float> target_probs(draft_tokens.size());

    float max_draft = draft_data[0];
    float max_target = target_data[0];
    for (size_t i = 1; i < draft_tokens.size(); ++i) {
        max_draft = std::max(max_draft, draft_data[i]);
        max_target = std::max(max_target, target_data[i]);
    }

    float draft_sum = 0.0f;
    float target_sum = 0.0f;
    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        draft_probs[i] = std::exp(draft_data[i] - max_draft);
        target_probs[i] = std::exp(target_data[i] - max_target);
        draft_sum += draft_probs[i];
        target_sum += target_probs[i];
    }

    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        draft_probs[i] /= draft_sum;
        target_probs[i] /= target_sum;
    }

    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        float draft_prob = draft_probs[i];
        float target_prob = target_probs[i];

        float acceptance = std::fmin(1.0f, target_prob / draft_prob);

        bool accepted = acceptance >= config_.acceptance_threshold;

        DraftToken dt{
            .token_id = draft_tokens[i],
            .draft_prob = draft_prob,
            .target_prob = target_prob,
            .accepted = accepted
        };
        result.tokens.push_back(dt);

        if (accepted) {
            result.num_accepted++;
        }

        logprob_tracker_.record(draft_tokens[i], std::log(target_prob), std::log(draft_prob), accepted);
    }

    result.kl_divergence = logprob_tracker_.compute_average_kl_divergence();

    return result;
}

float SpeculativeDecodingRunner::compute_kl_divergence(
    const std::vector<DraftToken>& tokens,
    const memory::Buffer<float>& draft_logits,
    const memory::Buffer<float>& target_logits
) const {
    (void)draft_logits;
    (void)target_logits;

    float kl = 0.0f;
    for (const auto& token : tokens) {
        if (token.draft_prob > 0.0f && token.target_prob > 0.0f) {
            kl += token.target_prob * std::log(token.target_prob / token.draft_prob);
        }
    }
    return kl;
}

void SpeculativeDecodingRunner::apply_tree_attention_mask(
    int num_draft_tokens,
    const stream::Stream& stream
) {
    (void)num_draft_tokens;
    (void)stream;
}

std::vector<int> SpeculativeDecodingRunner::decode(
    const memory::Buffer<float>& prompt_embeddings,
    int prompt_length,
    const stream::Stream& stream,
    std::function<void(memory::Buffer<float>&, const std::vector<int64_t>&, bool, const stream::Stream&)>
        forward_fn
) {
    logprob_tracker_.clear();
    draft_tokens_ = generate_draft_tokens(prompt_embeddings, prompt_length, stream, forward_fn);

    snapshot_kv_state();

    memory::Buffer<float> draft_logits(512);
    memory::Buffer<float> target_logits(512);

    forward_fn(draft_logits, {0}, false, stream);
    forward_fn(target_logits, {0}, false, stream);

    auto result = verify_draft_tokens(draft_tokens_, draft_logits, target_logits, stream);

    if (result.num_accepted < result.tokens.size()) {
        rollback_kv_state();
    } else {
        commit_kv_state();
    }

    std::vector<int> output_tokens;
    for (const auto& token : result.tokens) {
        if (token.accepted) {
            output_tokens.push_back(token.token_id);
        }
    }

    return output_tokens;
}

}  // namespace cuda::inference
