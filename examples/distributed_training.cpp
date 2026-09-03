/**
 * @file distributed_training.cpp
 * @brief Multi-GPU distributed all-reduce demo (current cuda::distributed API)
 *
 * Ported from the pre-v2 nova::nccl::* API (initialize(MPI_COMM_WORLD),
 * local_rank(), and root-rank all_reduce) that no longer exists. The current
 * library is device-mesh based: the shared cuda::nccl::NcclContext singleton
 * discovers the visible GPUs, and cuda::distributed::DistributedReduce runs a
 * real NCCL collective across the whole group.
 *
 * The collective contract requires one thread per device, each pinned with
 * cudaSetDevice, with every rank entering the operation together. This example
 * follows exactly that shape: each rank owns a distinct local chunk filled
 * with its device id, runs all_reduce(Sum), and verifies that every rank ends
 * with the group's total.
 *
 * Run (needs >= 2 visible GPUs):
 *   CUDA_VISIBLE_DEVICES=1,2 ./build/bin/distributed_training --chunk 1024
 *
 * On a single visible GPU the collective degenerates to a local copy and the
 * demo still verifies the identity result, so it is safe to run anywhere.
 */

#include <cuda/distributed/reduce.h>
#include <cuda/memory/buffer.h>
#include <cuda/memory/buffer-inl.h>
#include <cuda/nccl/nccl_context.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using cuda::memory::Buffer;

namespace {

struct Args {
    int chunk = 256;
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -h, --help    Show this help and exit\n");
    printf("  --chunk <n>   Number of floats each rank contributes (positive int)\n");
}

// Returns 1 on --help (exit 0), 0 on success, -1 on error (exit 1).
int parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v <= 0 || v > 1e9) {
                fprintf(stderr, "Error: --chunk expects a positive integer, got '%s'\n",
                        argv[i]);
                return -1;
            }
            args.chunk = static_cast<int>(v);
        } else {
            fprintf(stderr, "Error: unknown or incomplete option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    const int parse = parse_args(argc, argv, args);
    if (parse < 0) return 1;
    if (parse > 0) return 0;

    int device_count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count < 1) {
        fprintf(stderr, "Error: no CUDA-capable device available\n");
        return 1;
    }

    printf("Nova Distributed Training Example\n");
    printf("Visible devices: %d, Chunk: %d floats/rank\n", device_count, args.chunk);

    // Initialise the shared NCCL context for the whole device mesh before any
    // rank thread starts, so the per-rank collectives share one communicator
    // set (the same readiness step the multi-GPU suites use).
    auto& ctx = cuda::nccl::NcclContext::instance();
    try {
        ctx.initialize();
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: failed to initialise NCCL context: %s\n", e.what());
        return 1;
    }

    std::atomic<bool> all_ok{true};
    std::mutex error_mutex;
    std::vector<std::string> failures;
    std::vector<std::thread> workers;

    for (int device = 0; device < device_count; ++device) {
        workers.emplace_back([&, device]() {
            try {
                CUDA_CHECK(cudaSetDevice(device));

                // Distinct local chunk per rank: every element equals device+1,
                // so the group sum of all ranks is the expected result.
                std::vector<float> local(args.chunk, static_cast<float>(device + 1));
                Buffer<float> send_buf(args.chunk);
                Buffer<float> recv_buf(args.chunk);
                send_buf.copy_from(local.data(), args.chunk);

                cuda::distributed::DistributedReduce::all_reduce(
                    send_buf.data(), recv_buf.data(), args.chunk,
                    cuda::distributed::ReductionOp::Sum);

                float expected = 0.0f;
                for (int d = 0; d < device_count; ++d) {
                    expected += static_cast<float>(d + 1);
                }

                std::vector<float> reduced(args.chunk);
                recv_buf.copy_to(reduced.data(), args.chunk);
                for (float value : reduced) {
                    if (std::fabs(value - expected) > 1e-4f) {
                        all_ok.store(false);
                        std::lock_guard<std::mutex> lock(error_mutex);
                        failures.push_back(
                            "rank " + std::to_string(device) + ": element " +
                            std::to_string(value) + " != expected " +
                            std::to_string(expected));
                        return;
                    }
                }
                std::lock_guard<std::mutex> lock(error_mutex);
                printf("[rank %d] all_reduce OK: every element == %.1f\n",
                       device, expected);
            } catch (const std::exception& e) {
                all_ok.store(false);
                std::lock_guard<std::mutex> lock(error_mutex);
                failures.push_back("rank " + std::to_string(device) + ": " + e.what());
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (!all_ok.load()) {
        fprintf(stderr, "Distributed training FAILED:\n");
        for (const auto& failure : failures) {
            fprintf(stderr, "  %s\n", failure.c_str());
        }
        return 1;
    }
    printf("Distributed training complete: %d/%d ranks consistent\n",
           device_count, device_count);
    return 0;
}
