# Phase 72 Plan: Sequence Manager & Scheduler

## Goal

Implement multi-sequence orchestration with continuous batching and GQA/MQA support.

## Requirements

- SCHED-01: Multi-sequence management
- SCHED-02: Continuous batching with iteration-level scheduling
- SCHED-03: GQA/MQA support

## Implementation

### 1. Create Scheduler Header

**File:** `include/cuda/inference/scheduler.h`

```cpp
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
    int num_kv_heads = 8;  // For GQA/MQA
    int head_dim = 128;
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
    SequenceState get_state(int64_t seq_id) const;

    size_t get_num_active_sequences() const;

private:
    SchedulerConfig config_;
    std::unordered_map<int64_t, SequenceState> states_;
    std::unordered_map<int64_t, int> sequence_lengths_;
    mutable std::shared_mutex mutex_;
    int64_t next_seq_id_ = 0;
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

    SchedulerConfig config() const { return config_; }

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

}  // namespace cuda::inference
```

### 2. Implement SequenceManager

**File:** `src/cuda/inference/scheduler.cpp`

- add_sequence(): Allocate sequence with BlockManager, set state to Running
- update_sequence(): Update length tracking
- complete_sequence(): Mark finished, release resources
- remove_sequence(): Remove from tracking
- get_running_sequences(): Return sequences in Running state

### 3. Implement Scheduler

**File:** `src/cuda/inference/scheduler.cpp` (continued)

- add_request(): Create new sequence, add to pending
- get_batch(): Return current batch for iteration
- on_token_generated(): Update sequence length, check completion
- on_sequence_complete(): Mark finished, trigger next batch
- step(): Recompose batch if continuous batching enabled
- recompose_batch(): Add pending, remove finished, respect max batch size

### 4. GQA/MQA Support

- Pass num_kv_heads to attention config
- FlashAttention handles GQA internally (already implemented)
- KVCacheAllocator configured for num_kv_heads

### 5. Create Tests

**File:** `tests/inference/scheduler_test.cpp`

- Sequence lifecycle (create, update, complete, remove)
- Batch composition and size limits
- GQA/MQA configuration
- Continuous batching loop

## Files to Create/Modify

1. `include/cuda/inference/scheduler.h` (new)
2. `src/cuda/inference/scheduler.cpp` (new)
3. `tests/inference/scheduler_test.cpp` (new)
4. Update `include/cuda/inference/block_manager.h` (add forward declaration)

## Success Criteria

1. Multiple sequences coexist with independent state
2. Batched forward processes variable-length sequences
3. GQA/MQA produces correct output with num_kv_heads < num_q_heads
4. New sequences can be added without blocking
5. Completed sequences release resources
