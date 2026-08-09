#include "cuda/inference/block_manager.h"
#include "cuda/device/error.h"
#include <algorithm>
#include <chrono>

namespace cuda::inference {

BlockManager::BlockManager(const BlockManagerConfig& config)
    : config_(config) {

    max_blocks_per_sequence_ = (config_.max_model_len + config_.block_size - 1) /
                                config_.block_size;

    // num_gpu_blocks is the single source of truth for the device KV block
    // count. Propagate it into the allocator config so the two knobs cannot
    // silently disagree (e.g. a small num_gpu_blocks with the default
    // kv_cache_config.num_blocks would request far more GPU memory than the
    // manager can ever hand out).
    auto kv_config = config_.kv_cache_config;
    kv_config.num_blocks = config_.num_gpu_blocks;
    kv_cache_ = std::make_unique<memory::KVCacheAllocator>(kv_config);

    attention_ = algo::create_flash_attention(config_.attention_config);

    const size_t block_table_size = static_cast<size_t>(config_.num_cpu_blocks) *
                                     max_blocks_per_sequence_;
    block_table_gpu_ = memory::Buffer<int>(block_table_size);

    CUDA_CHECK(cudaEventCreate(&last_update_event_));
}

BlockManager::~BlockManager() {
    if (last_update_event_) {
        cudaEventDestroy(last_update_event_);
    }
}

Sequence* BlockManager::create_sequence(int64_t sequence_id, int max_tokens) {
    std::unique_lock lock(sequence_mutex_);

    if (sequences_.contains(sequence_id)) {
        throw std::runtime_error("Sequence " + std::to_string(sequence_id) +
                                 " already exists");
    }

    auto seq = std::make_unique<Sequence>(sequence_id, max_tokens);
    seq->created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();

    const int num_blocks = (max_tokens + config_.block_size - 1) /
                           config_.block_size;
    seq->block_table.resize(num_blocks, -1);

    Sequence* seq_ptr = seq.get();
    sequences_[sequence_id] = std::move(seq);
    sequence_to_index_[sequence_id] = num_allocated_sequences_;
    num_allocated_sequences_++;

    allocate_blocks_for_sequence(seq_ptr, num_blocks);

    return seq_ptr;
}

void BlockManager::append_tokens(int64_t sequence_id, int num_tokens) {
    std::unique_lock lock(sequence_mutex_);

    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw std::runtime_error("Sequence " + std::to_string(sequence_id) +
                                 " not found");
    }

    Sequence* seq = it->second.get();
    const int new_total = seq->num_tokens + num_tokens;

    if (new_total > seq->max_tokens) {
        throw std::runtime_error("Sequence " + std::to_string(sequence_id) +
                                 " exceeds max_tokens");
    }

    const int current_blocks = static_cast<int>(seq->block_table.size());
    const int needed_blocks = (new_total + config_.block_size - 1) /
                              config_.block_size;

    if (needed_blocks > current_blocks) {
        const int additional_blocks = needed_blocks - current_blocks;
        seq->block_table.resize(needed_blocks, -1);
        allocate_blocks_for_sequence(seq, additional_blocks);
    }

    seq->num_tokens = new_total;
}

Sequence* BlockManager::get_sequence(int64_t sequence_id) {
    std::shared_lock lock(sequence_mutex_);
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return nullptr;
    }
    return it->second.get();
}

const Sequence* BlockManager::get_sequence(int64_t sequence_id) const {
    std::shared_lock lock(sequence_mutex_);
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return nullptr;
    }
    return it->second.get();
}

void BlockManager::free_sequence(int64_t sequence_id) {
    std::unique_lock lock(sequence_mutex_);

    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return;
    }

    kv_cache_->free(sequence_id);
    sequence_to_index_.erase(sequence_id);
    sequences_.erase(it);
    num_allocated_sequences_--;
}

void BlockManager::forward_batch(
    const std::vector<int64_t>& sequence_ids,
    const memory::Buffer<float>& query,
    memory::Buffer<float>& output,
    const stream::Stream& stream
) {
    std::shared_lock lock(sequence_mutex_);

    for (const int64_t seq_id : sequence_ids) {
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) {
            throw std::runtime_error("Sequence " + std::to_string(seq_id) +
                                     " not found during forward_batch");
        }

        Sequence* seq = it->second.get();
        update_block_table_gpu(seq, stream);
    }

    sync_block_tables(stream);

    memory::Buffer<float> softmax_lse(config_.attention_config.num_heads *
                                      sequence_ids.size());

    attention_->forward(output, softmax_lse, query, query, query, stream);
}

void BlockManager::sync_block_tables(const stream::Stream& stream) {
    if (last_update_event_) {
        CUDA_CHECK(cudaStreamWaitEvent(stream.get(), last_update_event_));
    }
}

void BlockManager::maybe_evict() {
    std::unique_lock lock(sequence_mutex_);

    const int free_blocks = kv_cache_->get_num_free_blocks();
    const int threshold = config_.num_gpu_blocks *
                          config_.kv_cache_config.eviction_threshold_pct / 100;

    if (free_blocks < threshold) {
        const int blocks_needed = threshold - free_blocks;
        kv_cache_->evict(blocks_needed);
    }
}

int BlockManager::get_num_free_blocks() const {
    return kv_cache_->get_num_free_blocks();
}

std::vector<std::pair<int64_t, Sequence*>> BlockManager::get_active_sequences() const {
    std::shared_lock lock(sequence_mutex_);
    std::vector<std::pair<int64_t, Sequence*>> result;
    result.reserve(sequences_.size());
    for (const auto& [id, seq] : sequences_) {
        result.emplace_back(id, seq.get());
    }
    return result;
}

void BlockManager::allocate_blocks_for_sequence(Sequence* seq, int num_blocks) {
    auto blocks = kv_cache_->allocate(seq->id, num_blocks * config_.block_size);

    if (blocks.size() < static_cast<size_t>(num_blocks)) {
        throw std::runtime_error("Failed to allocate " +
                                 std::to_string(num_blocks) +
                                 " blocks for sequence " +
                                 std::to_string(seq->id));
    }

    for (int i = 0; i < num_blocks; ++i) {
        if (blocks[i] == nullptr) {
            throw std::runtime_error("Allocated block is null at index " +
                                     std::to_string(i));
        }
        seq->block_table[i] = blocks[i]->block_id;
    }
}

void BlockManager::validate_block_index(int block_idx) const {
    if (block_idx < 0 || block_idx >= config_.num_gpu_blocks) {
        throw std::out_of_range("Block index " + std::to_string(block_idx) +
                                " out of bounds [0, " +
                                std::to_string(config_.num_gpu_blocks) + ")");
    }
}

void BlockManager::update_block_table_gpu(
    Sequence* seq,
    const stream::Stream& stream
) {
    auto it = sequence_to_index_.find(seq->id);
    if (it == sequence_to_index_.end()) {
        throw std::runtime_error("Sequence index not found for ID " +
                                 std::to_string(seq->id));
    }

    const int seq_index = it->second;
    if (seq_index >= config_.num_cpu_blocks) {
        throw std::out_of_range("Sequence index " + std::to_string(seq_index) +
                                " exceeds allocated block table size");
    }

    const int seq_offset = seq_index * max_blocks_per_sequence_;
    const int* h_table = seq->block_table.data();
    const int num_blocks = static_cast<int>(seq->block_table.size());

    CUDA_CHECK(cudaMemcpyAsync(
        block_table_gpu_.data() + seq_offset,
        h_table,
        num_blocks * sizeof(int),
        cudaMemcpyHostToDevice,
        stream.get()
    ));

    CUDA_CHECK(cudaEventRecord(last_update_event_, stream.get()));
}

}  // namespace cuda::inference
