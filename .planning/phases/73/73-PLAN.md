# Phase 73 Plan: Sequence Parallelism Extension

## Goal

Implement distributed attention computation across tensor parallel ranks for long context support.

## Requirements

- SP-01: SequenceParallelAttention across TP ranks
- SP-02: Ring sequence parallelism for long sequences
- SP-03: TP communicator integration

## Implementation

### 1. Create Sequence Parallel Header

**File:** `include/cuda/distributed/sequence_parallel.h`

```cpp
namespace cuda::distributed {

struct SequenceParallelConfig {
    int num_model_parallel_gpus = 1;
    int sequence_parallel_size = 1;
    bool reduce_scatter_output = true;
    ncclComm_t comm = nullptr;
};

class SequenceParallelAttention {
public:
    explicit SequenceParallelAttention(const SequenceParallelConfig& config);
    ~SequenceParallelAttention();

    void gather_kv(
        memory::Buffer<float>& gathered_k,
        memory::Buffer<float>& gathered_v,
        const memory::Buffer<float>& local_k,
        const memory::Buffer<float>& local_v
    );

    void scatter_output(
        memory::Buffer<float>& local_output,
        const memory::Buffer<float>& full_output
    );

    void all_reduce_sequence(memory::Buffer<float>& data);

    bool has_sequence_parallelism() const {
        return config_.sequence_parallel_size > 1;
    }

private:
    SequenceParallelConfig config_;
};

class RingSequenceParallelism {
public:
    explicit RingSequenceParallelism(const SequenceParallelConfig& config);
    ~RingSequenceParallelism();

    void ring_attention(
        memory::Buffer<float>& query,
        memory::Buffer<float>& key,
        memory::Buffer<float>& value,
        memory::Buffer<float>& output
    );

private:
    SequenceParallelConfig config_;
};

}  // namespace cuda::distributed
```

### 2. Implement SequenceParallelAttention

**File:** `src/cuda/distributed/sequence_parallel.cpp`

- gather_kv(): All-gather KV across sequence parallel ranks
- scatter_output(): Reduce-scatter output
- all_reduce_sequence(): All-reduce for non-attention layers

### 3. Implement RingSequenceParallelism

**File:** `src/cuda/distributed/sequence_parallel.cpp` (continued)

- ring_attention(): Pass KV around ring, accumulate attention
- Use send/recv for KV projection communication

### 4. Integration with Existing TP

**File:** Extend `cuda/distributed/matmul.h`

- Add sequence_parallel_size to TensorParallelMatmul
- Get sequence parallel communicator
- Integrate with attention backward pass

### 5. Single-GPU Fallback

- Check sequence_parallel_size == 1
- Skip all communication if single-GPU
- Return input as output unchanged

### 6. Create Tests

**File:** `tests/distributed/sequence_parallel_test.cpp`

- Single-GPU fallback
- Multi-GPU configuration
- KV gathering and output scattering

## Files to Create

1. `include/cuda/distributed/sequence_parallel.h`
2. `src/cuda/distributed/sequence_parallel.cpp`
3. `tests/distributed/sequence_parallel_test.cpp`

## Success Criteria

1. Sequence attention output across TP ranks matches single-GPU result
2. Ring sequence parallelism handles sequences up to 128K tokens
3. TP communicator correctly reduces sequence parallel output
4. Ring attention communicates KV projections with minimal sync
5. Sequence parallelism disables gracefully on single-GPU
