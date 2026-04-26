/**
 * @file distributed_training.cpp
 * @brief Distributed training example with NCCL collectives
 * @example
 *
 * Compile:
 *   mpicc -std=c++23 -I include examples/distributed_training.cpp \
 *       -L build/lib -lcuda_impl -lcudart -lnccl -o distributed_training
 *
 * Run (2 GPUs):
 *   mpirun -n 2 --allow-run-as-root ./distributed_training --batch 64 --epochs 10
 */

#include <cuda/nccl/nccl_context.hpp>
#include <cuda/distributed/reduce.hpp>
#include <cuda/distributed/all_gather.hpp>
#include <cuda/neural/matmul.hpp>
#include <cuda/neural/softmax.hpp>
#include <cuda/error/cuda_error.hpp>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

struct Args {
    int batch = 64;
    int hidden = 512;
    int epochs = 10;
    float learning_rate = 0.001f;
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --batch <n>         Batch size\n");
    printf("  --hidden <n>        Hidden dimension\n");
    printf("  --epochs <n>        Number of epochs\n");
    printf("  --learning_rate <f> Learning rate\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            args.batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            args.hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            args.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learning_rate") == 0 && i + 1 < argc) {
            args.learning_rate = atof(argv[++i]);
        } else {
            print_usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Nova Distributed Training Example\n");
        printf("World size: %d, Batch: %d, Hidden: %d, Epochs: %d\n",
               size, args.batch, args.hidden, args.epochs);
    }

    // Initialize NCCL
    nova::nccl::NcclContext::initialize(MPI_COMM_WORLD);
    auto& nccl_ctx = nova::nccl::NcclContext::instance();

    int local_rank = nccl_ctx.local_rank();
    cudaSetDevice(local_rank);

    const int N = args.batch;
    const int K = args.hidden;
    const int M = args.hidden;

    if (rank == 0) {
        printf("Rank %d using GPU %d\n", rank, local_rank);
    }

    // Allocate local data
    nova::memory::Buffer<float> local_input(N * K);
    nova::memory::Buffer<float> local_grad(N * M);
    nova::memory::Buffer<float> local_output(N * M);

    // Initialize with random values
    std::mt19937 rng(rank * 42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < local_input.size(); i++) {
        local_input.host_data()[i] = dist(rng);
    }
    local_input.sync_to_device();

    // Training loop
    for (int epoch = 0; epoch < args.epochs; epoch++) {
        // Forward pass
        nova::neural::matmul(
            local_input.device_data(),
            nullptr,  // weight (identity for demo)
            local_output.device_data(),
            N, K, M
        );

        nova::neural::relu(local_output.device_data(),
                          local_output.device_data(), N * M);

        // All-reduce gradients
        nova::distributed::all_reduce(
            local_grad.device_data(),
            N * M,
            0,  // root rank
            nccl_ctx.stream()
        );

        if (rank == 0 && epoch % 5 == 0) {
            printf("Epoch %d/%d complete\n", epoch + 1, args.epochs);
        }
    }

    // Synchronize
    cudaStreamSynchronize(nccl_ctx.stream());

    if (rank == 0) {
        printf("Distributed training complete!\n");
    }

    nova::nccl::NcclContext::shutdown();
    MPI_Finalize();
    return 0;
}
