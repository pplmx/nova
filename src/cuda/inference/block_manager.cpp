#include "cuda/inference/block_manager.h"
#include "cuda/device/error.h"
#include <algorithm>
#include <chrono>

namespace cuda::inference {

BlockManager::BlockManager(const BlockManagerConfig& config)
    : config_(config) {

    max_blocks_per_sequence_ = (config_.max_model_len + config_.block_size - 1) /
                                config_.block_size;

    // num_gpu_blocks (the manager's hand-out pool) and kv_cache_config.num_blocks
    // (the allocator's pool) are intentionally distinct knobs: the allocator's
    // block pool backs the KV cache and is configured independently. Coupling
    // them made the allocator pool equal the (often small) scheduling count,
    // breaking long-prompt tests that pre-allocate more blocks than
    // num_gpu_blocks advertises. RIL Round 9 misjudged this as "two knobs that
    // silently disagree"; the tests are the contract. (commit 75913f5 reintroduced
    // mismatches; reverted here.)
    kv_cache_ = std::make_unique<memory::KVCacheAllocator>(config_.kv_cache_config);

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

    // Allocate the KV blocks BEFORE committing the sequence to the maps: on a
    // throw, roll back the partial KV registration so no index slot is burned
    // and no sequence id is left half-created in sequences_ (review note).
    try {
        allocate_blocks_for_sequence(seq.get(), num_blocks);
    } catch (...) {
        kv_cache_->free(sequence_id);
        throw;
    }

    Sequence* seq_ptr = seq.get();
    sequences_[sequence_id] = std::move(seq);
    // Reuse freed block-table slots before handing out a fresh index (RIL
    // TASK-079, ISS-019): the old ever-increasing counter exhausted the
    // num_cpu_blocks-sized GPU table after that many cumulative create_sequence
    // calls, after which every forward_batch threw out_of_range.
    if (!free_indices_.empty()) {
        sequence_to_index_[sequence_id] = free_indices_.back();
        free_indices_.pop_back();
    } else {
        sequence_to_index_[sequence_id] = num_allocated_sequences_++;
    }

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

    // Return the slot to the reuse pool. NOTE: num_allocated_sequences_ is the
    // monotonic "next fresh index" cursor and must NOT be decremented here —
    // decrementing it would re-hand out an index that is still in the free
    // pool, double-booking the same block-table row.
    auto idx_it = sequence_to_index_.find(sequence_id);
    if (idx_it != sequence_to_index_.end()) {
        free_indices_.push_back(idx_it->second);
    }
    kv_cache_->free(sequence_id);
    sequence_to_index_.erase(sequence_id);
    sequences_.erase(it);
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
