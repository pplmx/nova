/**
 * @file neural_net.cpp
 * @brief Neural network primitives example demonstrating matmul, softmax, and activations
 * @example
 *
 * Compile:
 *   g++ -std=c++23 -I include examples/neural_net.cpp \
 *       -L build/lib -lcuda_impl -lcudart -lcublas -o neural_net
 *
 * Run:
 *   ./neural_net --batch 32 --seq_len 128 --hidden 512
 */

#include <cuda/neural/matmul.hpp>
#include <cuda/neural/softmax.hpp>
#include <cuda/neural/activations.hpp>
#include <cuda/neural/layer_norm.hpp>
#include <cuda/error/cuda_error.hpp>
#include <cstdio>
#include <cstdlib>
#include <random>

struct Args {
    int batch = 32;
    int seq_len = 128;
    int hidden = 512;
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --batch <n>     Batch size\n");
    printf("  --seq_len <n>   Sequence length\n");
    printf("  --hidden <n>    Hidden dimension\n");
}

int main(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            args.batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq_len") == 0 && i + 1 < argc) {
            args.seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            args.hidden = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("Nova Neural Net Primitives Example\n");
    printf("Batch: %d, SeqLen: %d, Hidden: %d\n",
           args.batch, args.seq_len, args.hidden);

    const int N = args.batch * args.seq_len;
    const int K = args.hidden;
    const int M = args.hidden;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    // Allocate buffers
    nova::memory::Buffer<float> input(N * K);
    nova::memory::Buffer<float> weight(K * M);
    nova::memory::Buffer<float> bias(M);
    nova::memory::Buffer<float> output(N * M);
    nova::memory::Buffer<float> temp(N * M);

    // Initialize with random values
    for (size_t i = 0; i < input.size(); i++) {
        input.host_data()[i] = dist(rng);
    }
    for (size_t i = 0; i < weight.size(); i++) {
        weight.host_data()[i] = dist(rng);
    }
    for (size_t i = 0; i < bias.size(); i++) {
        bias.host_data()[i] = 0.0f;
    }

    input.sync_to_device();
    weight.sync_to_device();
    bias.sync_to_device();

    // 1. Matrix Multiply with Bias
    printf("Running matmul with bias...\n");
    nova::neural::matmul_bias(
        input.device_data(),   // [N, K]
        weight.device_data(),  // [K, M]
        bias.device_data(),    // [M]
        output.device_data(),  // [N, M]
        N, K, M
    );

    // 2. Apply ReLU activation
    printf("Applying ReLU...\n");
    nova::neural::relu(output.device_data(), output.device_data(), N * M);

    // 3. Apply Layer Normalization
    printf("Applying LayerNorm...\n");
    nova::neural::layer_norm(
        output.device_data(),  // input [N, M]
        temp.device_data(),    // output [N, M]
        N, M,                  // [N, M] matrix
        1e-5f                  // epsilon
    );

    // 4. Apply Softmax
    printf("Applying Softmax...\n");
    for (int b = 0; b < args.batch; b++) {
        int offset = b * args.seq_len * args.hidden;
        nova::neural::softmax(
            temp.device_data() + offset,
            temp.device_data() + offset,
            args.seq_len,    // rows
            args.hidden      // cols
        );
    }

    temp.sync_to_host();

    // Verify output is valid
    float min_val = 1e9, max_val = -1e9, sum = 0.0f;
    for (int i = 0; i < std::min(100, N * M); i++) {
        float v = temp.host_data()[i];
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
    }

    printf("Output statistics (first 100 elements):\n");
    printf("  Min: %.4f, Max: %.4f\n", min_val, max_val);
    printf("  Sum: %.4f (should be ~batch for softmax)\n", sum);

    // Check softmax constraint (each row should sum to 1)
    printf("\nVerifying softmax outputs...\n");
    bool valid = true;
    for (int b = 0; b < std::min(3, args.batch); b++) {
        float row_sum = 0.0f;
        for (int s = 0; s < std::min(10, args.seq_len); s++) {
            row_sum += temp.host_data()[b * args.seq_len * args.hidden +
                                    s * args.hidden];
        }
        printf("  Batch %d first 10 seqs sum: %.4f\n", b, row_sum);
        if (std::abs(row_sum - 10.0f) > 0.1f) valid = false;
    }

    if (valid) {
        printf("\nAll checks passed! Neural net forward pass successful.\n");
    } else {
        printf("\nWarning: Some outputs look incorrect.\n");
    }

    return 0;
}
