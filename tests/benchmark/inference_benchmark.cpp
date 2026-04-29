#include <benchmark/benchmark.h>
#include <cuda/inference/scheduler.h>
#include <cuda/stream/stream.h>
#include <cuda/device/error.h>
#include <chrono>
#include <vector>

namespace cuda::inference::benchmark {

static void BM_Throughput(benchmark::State& state) {
    CUDA_CHECK(cudaSetDevice(0));
    stream::Stream stream;

    SchedulerConfig config{
        .max_batch_size = state.range(0),
        .max_sequence_length = 512,
        .num_heads = 8,
        .num_kv_heads = 8,
        .head_dim = 64,
        .block_size = 16
    };

    auto scheduler = std::make_unique<Scheduler>(config);

    for (int i = 0; i < state.range(0); ++i) {
        scheduler->add_request(64);
    }

    for (auto _ : state) {
        auto batch = scheduler->get_batch();

        memory::Buffer<float> query(64 * 8 * 64);
        memory::Buffer<float> output(64 * 8 * 64);

        scheduler->forward_batch(query, output, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        scheduler->step();
    }

    state.SetItemsProcessed(state.iterations() * state.range(0));
}

BENCHMARK(BM_Throughput)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Unit(benchmark::kMillisecond);

static void BM_KVCacheMemoryEfficiency(benchmark::State& state) {
    CUDA_CHECK(cudaSetDevice(0));
    stream::Stream stream;

    SchedulerConfig config{
        .max_batch_size = state.range(0),
        .max_sequence_length = 512,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .block_size = 16
    };

    auto scheduler = std::make_unique<Scheduler>(config);

    for (int i = 0; i < state.range(0); ++i) {
        scheduler->add_request(256);
    }

    auto* kv_cache = scheduler->get_block_manager().get_kv_cache();
    auto stats = kv_cache->get_stats();

    const int tokens_used = state.range(0) * 256;
    const int blocks_used = stats.allocated_blocks;
    const int ideal_blocks = (tokens_used + config.block_size - 1) / config.block_size;
    const float waste_percent = 100.0f * (blocks_used - ideal_blocks) / blocks_used;

    state.SetLabel("Waste: " + std::to_string(waste_percent) + "%");

    for (auto _ : state) {
        benchmark::DoNotOptimize(stats);
    }
}

BENCHMARK(BM_KVCacheMemoryEfficiency)
    ->RangeMultiplier(2)
    ->Range(1, 16)
    ->Unit(benchmark::kMillisecond);

static void BM_LatencyPerToken(benchmark::State& state) {
    CUDA_CHECK(cudaSetDevice(0));
    stream::Stream stream;

    SchedulerConfig config{
        .max_batch_size = 1,
        .max_sequence_length = 512,
        .num_heads = 8,
        .num_kv_heads = 8,
        .head_dim = 64,
        .block_size = 16
    };

    auto scheduler = std::make_unique<Scheduler>(config);
    int64_t seq_id = scheduler->add_request(64);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        auto batch = scheduler->get_batch();
        memory::Buffer<float> query(64 * 8 * 64);
        memory::Buffer<float> output(64 * 8 * 64);

        scheduler->forward_batch(query, output, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        scheduler->get_block_manager().append_tokens(seq_id, 1);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        state.SetIterationTime(duration.count() / 1e6);
    }
}

BENCHMARK(BM_LatencyPerToken)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1);

BENCHMARK_MAIN();

}  // namespace cuda::inference::benchmark
