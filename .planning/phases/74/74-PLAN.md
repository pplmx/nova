# Phase 74 Plan: Integration & Testing

## Goal

End-to-end validation with CUDA Graphs, observability, and performance benchmarks.

## Requirements

All v2.6 requirements from previous phases

## Implementation

### 1. Create NVTX Domain for Inference

**File:** `include/cuda/observability/inference_nvtx.h`

```cpp
namespace cuda::observability {

class InferenceNVTXDomain {
public:
    static InferenceNVTXDomain& get();

    void begin_prefill();
    void end_prefill();

    void begin_decode();
    void end_decode();

    void begin_attention(const char* name);
    void end_attention();

    void begin_scheduling();
    void end_scheduling();

    void record_batch_size(int size);
    void record_sequence_length(int length);

private:
    InferenceNVTXDomain();
    NVTXDomain domain_;
};

}  // namespace cuda::observability
```

### 2. Extend GraphExecutor for Inference

**File:** `include/cuda/production/inference_graph.h`

```cpp
namespace cuda::production {

class InferenceGraphExecutor : public GraphExecutor {
public:
    explicit InferenceGraphExecutor(const BlockManager& block_manager);

    void capture_inference_pass(
        const std::vector<int64_t>& sequence_ids,
        const memory::Buffer<float>& query,
        memory::Buffer<float>& output
    );

    void update_batch_size(int new_batch_size);

private:
    const BlockManager& block_manager_;
    int current_batch_size_ = 0;
};

}  // namespace cuda::production
```

### 3. Create Integration Test

**File:** `tests/inference/integration_test.cpp`

```cpp
TEST(IntegrationTest, EndToEndInference) {
    SchedulerConfig config = /* ... */;
    Scheduler scheduler(config);

    for (int i = 0; i < 10; ++i) {
        scheduler.add_request(64);
    }

    auto batch = scheduler.get_batch();
    ASSERT_FALSE(batch.empty());

    memory::Buffer<float> query(/* ... */);
    memory::Buffer<float> output(/* ... */);

    scheduler.forward_batch(query, output, *stream);
    CUDA_CHECK(cudaStreamSynchronize(stream->get()));
}
```

### 4. Create Performance Benchmark

**File:** `tests/benchmark/inference_benchmark.cpp`

- Throughput benchmark (sequences/second)
- Memory efficiency (KV cache waste %)
- Latency benchmark (ms per token)
- Multi-sequence scaling test

### 5. NVTX Integration with Scheduler

Update Scheduler to annotate phases:
- `begin_prefill()` on batch start
- `begin_attention()` during attention
- `end_decode()` after token generation

## Files to Create

1. `include/cuda/observability/inference_nvtx.h`
2. `include/cuda/production/inference_graph.h`
3. `tests/inference/integration_test.cpp`
4. `tests/benchmark/inference_benchmark.cpp`

## Success Criteria

1. CUDA Graph capture/replay works with dynamic block allocation
2. NVTX annotations mark inference phases
3. Throughput benchmark shows >2x speedup
4. Memory efficiency <4% KV cache waste
5. All 18 v2.6 requirements pass tests
