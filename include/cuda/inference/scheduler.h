#pragma once

#include "cuda/inference/block_manager.h"
#include "cuda/stream/stream.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace cuda::inference {

enum class SequenceState {
    Waiting,
    Running,
    Finished,
    Evicted
};

struct SchedulerConfig {
    int max_batch_size = 32;
    int max_sequence_length = 8192;
    int prefill_batch_size = 8;
    bool enable_continuous_batching = true;
    bool enable_prefix_caching = true;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    int block_size = 16;
};

class SequenceManager {
public:
    explicit SequenceManager(const SchedulerConfig& config);
    ~SequenceManager();

    int64_t add_sequence(int max_tokens);
    void update_sequence(int64_t seq_id, int new_length);
    void complete_sequence(int64_t seq_id);
    void remove_sequence(int64_t seq_id);

    std::vector<int64_t> get_running_sequences() const;
    std::vector<int64_t> get_waiting_sequences() const;
    std::vector<int64_t> get_finished_sequences() const;
    SequenceState get_state(int64_t seq_id) const;

    size_t get_num_active_sequences() const;
    size_t get_num_finished_sequences() const;

    SchedulerConfig config() const { return config_; }

private:
    SchedulerConfig config_;
    std::unordered_map<int64_t, SequenceState> states_;
    std::unordered_map<int64_t, int> sequence_lengths_;
    std::unordered_map<int64_t, int> sequence_max_tokens_;
    mutable std::shared_mutex mutex_;
    int64_t next_seq_id_ = 0;
    size_t num_active_ = 0;
    size_t num_finished_ = 0;
};

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler();

    int64_t add_request(int max_tokens);
    std::vector<int64_t> get_batch();
    void on_token_generated(int64_t seq_id);
    void on_sequence_complete(int64_t seq_id);
    void step();

    void forward_batch(
        const memory::Buffer<float>& query,
        memory::Buffer<float>& output,
        const stream::Stream& stream
    );

    SchedulerConfig config() const { return config_; }
    SequenceManager& get_sequence_manager() { return seq_manager_; }
    BlockManager& get_block_manager() { return block_manager_; }

private:
    void recompose_batch();
    bool should_preempt() const;

    SchedulerConfig config_;
    SequenceManager seq_manager_;
    BlockManager block_manager_;
    stream::Stream compute_stream_;

    std::vector<int64_t> active_batch_;
    std::vector<int64_t> pending_requests_;
    int current_batch_size_ = 0;
};

SequenceManager::SequenceManager(const SchedulerConfig& config)
    : config_(config), next_seq_id_(0), num_active_(0), num_finished_(0) {}

SequenceManager::~SequenceManager() = default;

int64_t SequenceManager::add_sequence(int max_tokens) {
    std::unique_lock lock(mutex_);

    const int64_t seq_id = next_seq_id_++;
    states_[seq_id] = SequenceState::Running;
    sequence_lengths_[seq_id] = 0;
    sequence_max_tokens_[seq_id] = max_tokens;
    num_active_++;

    return seq_id;
}

void SequenceManager::update_sequence(int64_t seq_id, int new_length) {
    std::unique_lock lock(mutex_);

    auto it = sequence_lengths_.find(seq_id);
    if (it != sequence_lengths_.end()) {
        it->second = new_length;
    }
}

void SequenceManager::complete_sequence(int64_t seq_id) {
    std::unique_lock lock(mutex_);

    auto it = states_.find(seq_id);
    if (it != states_.end() && it->second == SequenceState::Running) {
        it->second = SequenceState::Finished;
        num_active_--;
        num_finished_++;
    }
}

void SequenceManager::remove_sequence(int64_t seq_id) {
    std::unique_lock lock(mutex_);

    auto state_it = states_.find(seq_id);
    if (state_it != states_.end()) {
        if (state_it->second == SequenceState::Running) {
            num_active_--;
        } else if (state_it->second == SequenceState::Finished) {
            num_finished_--;
        }
        states_.erase(state_it);
    }

    sequence_lengths_.erase(seq_id);
    sequence_max_tokens_.erase(seq_id);
}

std::vector<int64_t> SequenceManager::get_running_sequences() const {
    std::shared_lock lock(mutex_);

    std::vector<int64_t> result;
    for (const auto& [seq_id, state] : states_) {
        if (state == SequenceState::Running) {
            result.push_back(seq_id);
        }
    }
    return result;
}

std::vector<int64_t> SequenceManager::get_waiting_sequences() const {
    std::shared_lock lock(mutex_);

    std::vector<int64_t> result;
    for (const auto& [seq_id, state] : states_) {
        if (state == SequenceState::Waiting) {
            result.push_back(seq_id);
        }
    }
    return result;
}

std::vector<int64_t> SequenceManager::get_finished_sequences() const {
    std::shared_lock lock(mutex_);

    std::vector<int64_t> result;
    for (const auto& [seq_id, state] : states_) {
        if (state == SequenceState::Finished) {
            result.push_back(seq_id);
        }
    }
    return result;
}

SequenceState SequenceManager::get_state(int64_t seq_id) const {
    std::shared_lock lock(mutex_);

    auto it = states_.find(seq_id);
    if (it == states_.end()) {
        return SequenceState::Evicted;
    }
    return it->second;
}

size_t SequenceManager::get_num_active_sequences() const {
    std::shared_lock lock(mutex_);
    return num_active_;
}

size_t SequenceManager::get_num_finished_sequences() const {
    std::shared_lock lock(mutex_);
    return num_finished_;
}

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config),
      seq_manager_(config),
      block_manager_(BlockManagerConfig{
          .max_model_len = config.max_sequence_length,
          .block_size = config.block_size,
          .num_gpu_blocks = config.max_batch_size * 64,
          .kv_cache_config{
              .num_heads = config.num_kv_heads,
              .head_dim = config.head_dim,
              .block_size_tokens = config.block_size,
              .num_blocks = config.max_batch_size * 64,
              .num_layers = 32
          },
          .attention_config{
              .num_heads = config.num_heads,
              .num_kv_heads = config.num_kv_heads,
              .head_dim = config.head_dim,
              .seq_len = config.max_sequence_length,
              .batch_size = config.max_batch_size
          }
      }) {}

Scheduler::~Scheduler() = default;

int64_t Scheduler::add_request(int max_tokens) {
    const int64_t seq_id = seq_manager_.add_sequence(max_tokens);
    block_manager_.create_sequence(seq_id, max_tokens);
    pending_requests_.push_back(seq_id);
    return seq_id;
}

std::vector<int64_t> Scheduler::get_batch() {
    if (config_.enable_continuous_batching) {
        recompose_batch();
    }
    return active_batch_;
}

void Scheduler::on_token_generated(int64_t seq_id) {
    auto length_it = seq_manager_.config().max_sequence_length;
    (void)length_it;

    auto seq = block_manager_.get_sequence(seq_id);
    if (seq) {
        seq_manager_.update_sequence(seq_id, seq->num_tokens + 1);

        if (seq->num_tokens >= seq->max_tokens) {
            on_sequence_complete(seq_id);
        }
    }
}

void Scheduler::on_sequence_complete(int64_t seq_id) {
    seq_manager_.complete_sequence(seq_id);

    auto it = std::find(active_batch_.begin(), active_batch_.end(), seq_id);
    if (it != active_batch_.end()) {
        active_batch_.erase(it);
    }

    if (config_.enable_continuous_batching) {
        recompose_batch();
    }
}

void Scheduler::step() {
    if (config_.enable_continuous_batching) {
        recompose_batch();
    }
}

void Scheduler::forward_batch(
    const memory::Buffer<float>& query,
    memory::Buffer<float>& output,
    const stream::Stream& stream
) {
    if (active_batch_.empty()) {
        return;
    }

    block_manager_.forward_batch(active_batch_, query, output, stream);
}

void Scheduler::recompose_batch() {
    std::vector<int64_t> to_remove;
    for (const int64_t seq_id : active_batch_) {
        const SequenceState state = seq_manager_.get_state(seq_id);
        if (state == SequenceState::Finished || state == SequenceState::Evicted) {
            to_remove.push_back(seq_id);
        }
    }

    for (const int64_t seq_id : to_remove) {
        auto it = std::find(active_batch_.begin(), active_batch_.end(), seq_id);
        if (it != active_batch_.end()) {
            active_batch_.erase(it);
        }
    }

    const int slots_available = config_.max_batch_size -
                                static_cast<int>(active_batch_.size());

    for (int i = 0; i < slots_available && !pending_requests_.empty(); ++i) {
        const int64_t next_seq = pending_requests_.front();
        pending_requests_.erase(pending_requests_.begin());

        if (seq_manager_.get_state(next_seq) == SequenceState::Running) {
            active_batch_.push_back(next_seq);
        }
    }

    current_batch_size_ = static_cast<int>(active_batch_.size());
}

bool Scheduler::should_preempt() const {
    return static_cast<int>(active_batch_.size()) >= config_.max_batch_size;
}

}  // namespace cuda::inference
