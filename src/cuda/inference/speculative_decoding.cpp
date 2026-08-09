#include "cuda/inference/speculative_decoding.h"
#include "cuda/inference/beam_search.h"
#include "cuda/device/error.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace cuda::inference {

namespace {

int sample_from_logits(
    const float* logits,
    int vocab_size,
    float temperature,
    uint64_t seed
) {
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    std::vector<float> probs(vocab_size);
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }

    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    std::vector<std::pair<int, float>> sorted(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        sorted[i] = {i, probs[i]};
    }
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    float cumsum = 0.0f;
    float target = static_cast<float>(seed % 10000) / 10000.0f;
    cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += sorted[i].second;
        if (cumsum >= target) {
            return sorted[i].first;
        }
    }

    return sorted[0].first;
}

}  // anonymous namespace

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
    kv_snapshot_ = KVCacheSnapshot{};
    kv_snapshot_->sequence_ids.clear();
    kv_snapshot_->num_blocks.clear();

    for (const auto& [seq_id, seq] : block_manager_->get_active_sequences()) {
        (void)seq;
        kv_snapshot_->sequence_ids.push_back(seq_id);
        kv_snapshot_->num_blocks.push_back(static_cast<int>(seq->block_table.size()));
    }
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

    const int vocab_size = config_.vocab_size > 0 ? config_.vocab_size : 32000;
    memory::Buffer<float> logits(vocab_size);

    std::mt19937 rng(static_cast<unsigned int>(seq_id));
    float temperature = config_.temperature > 0.0f ? config_.temperature : 0.8f;

    for (int step = 0; step < config_.draft_depth; ++step) {
        std::vector<int64_t> seq_ids = {seq_id};
        forward_fn(logits, seq_ids, false, stream);

        // sample_from_logits is a host function; the draft logits live on the
        // device, so stage a host copy before sampling (reading Buffer::data()
        // on the CPU would dereference device memory and segfault).
        std::vector<float> h_logits(vocab_size);
        logits.copy_to(h_logits.data(), vocab_size);
        uint64_t seed = static_cast<uint64_t>(step) * 1000 + rng();

        int token = sample_from_logits(h_logits.data(), vocab_size, temperature, seed);
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

    const int vocab_size = config_.vocab_size > 0 ? config_.vocab_size : 32000;

    // Use the actual logits buffer size: callers may pass buffers smaller than
    // config_.vocab_size (which defaults to a large value), and reading or
    // copying beyond the buffer is out-of-bounds.
    const size_t n = std::min(static_cast<size_t>(vocab_size),
                              std::min(draft_logits.size(), target_logits.size()));

    // The logits live on the device; stage host copies before the softmax
    // math below (reading Buffer::data() from the CPU segfaults).
    std::vector<float> h_draft(n);
    std::vector<float> h_target(n);
    draft_logits.copy_to(h_draft.data(), n);
    target_logits.copy_to(h_target.data(), n);
    const float* draft_data = h_draft.data();
    const float* target_data = h_target.data();

    if (n == 0) {
        return result;
    }

    float max_draft = draft_data[0];
    float max_target = target_data[0];
    for (size_t i = 1; i < n; ++i) {
        max_draft = std::max(max_draft, draft_data[i]);
        max_target = std::max(max_target, target_data[i]);
    }

    float draft_sum = 0.0f;
    float target_sum = 0.0f;
    std::vector<float> draft_exp(n);
    std::vector<float> target_exp(n);
    for (size_t i = 0; i < n; ++i) {
        draft_exp[i] = std::exp(draft_data[i] - max_draft);
        target_exp[i] = std::exp(target_data[i] - max_target);
        draft_sum += draft_exp[i];
        target_sum += target_exp[i];
    }

    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        int token_id = draft_tokens[i];
        if (token_id < 0 || static_cast<size_t>(token_id) >= n) {
            continue;
        }

        float draft_prob = draft_exp[token_id] / draft_sum;
        float target_prob = target_exp[token_id] / target_sum;

        constexpr float epsilon = 1e-8f;
        float acceptance;
        if (draft_prob > epsilon) {
            acceptance = std::fmin(1.0f, target_prob / draft_prob);
        } else {
            acceptance = 0.0f;
        }

        bool accepted = acceptance >= config_.acceptance_threshold;

        DraftToken dt{
            .token_id = token_id,
            .draft_prob = draft_prob,
            .target_prob = target_prob,
            .accepted = accepted
        };
        result.tokens.push_back(dt);

        if (accepted) {
            result.num_accepted++;
        }

        logprob_tracker_.record(token_id, std::log(target_prob + epsilon), std::log(draft_prob + epsilon), accepted);
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

    const int vocab_size = config_.vocab_size > 0 ? config_.vocab_size : 32000;
    memory::Buffer<float> draft_logits(vocab_size);
    memory::Buffer<float> target_logits(vocab_size);

    auto* verify_seq = block_manager_->create_sequence(0, prompt_length + config_.max_draft_depth);
    int64_t verify_seq_id = verify_seq->id;

    for (int i = 0; i < static_cast<int>(draft_tokens_.size()); ++i) {
        if (i < static_cast<int>(draft_tokens_.size()) - 1) {
            block_manager_->append_tokens(verify_seq_id, 1);
        }
    }

    std::vector<int64_t> seq_ids = {verify_seq_id};
    forward_fn(draft_logits, seq_ids, false, stream);

    block_manager_->append_tokens(verify_seq_id, static_cast<int>(draft_tokens_.size()));
    forward_fn(target_logits, seq_ids, false, stream);

    auto result = verify_draft_tokens(draft_tokens_, draft_logits, target_logits, stream);

    if (result.num_accepted < result.tokens.size()) {
        rollback_kv_state();
    } else {
        commit_kv_state();
    }

    block_manager_->free_sequence(verify_seq_id);

    std::vector<int> output_tokens;
    for (const auto& token : result.tokens) {
        if (token.accepted) {
            output_tokens.push_back(token.token_id);
        }
    }

    return output_tokens;
}

}  // namespace cuda::inference
