/**
 * @file neural_net.cpp
 * @brief Neural network primitives example: fused matmul+bias+ReLU, LayerNorm,
 *        and per-row softmax over the current cuda:: API (re-integrated for
 *        TASK-064; the pre-v2 nova::* API this example used no longer exists).
 * @example
 *
 * Compile:
 *   g++ -std=c++23 -I include examples/neural_net.cpp \
 *       -L build/lib -lcuda_impl -lcudart -lcublas -lcublasLt -o neural_net
 *
 * Run:
 *   ./neural_net --batch 32 --seq_len 128 --hidden 512
 */

#include <cuda/neural/matmul.h>
#include <cuda/neural/softmax.h>
#include <cuda/neural/activations.h>
#include <cuda/neural/layer_norm.h>
#include <cuda/neural/fusion/fused_matmul_bias_act.h>
#include <cuda/memory/buffer.h>
#include <cuda/memory/buffer-inl.h>
#include <cuda/device/error.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
using cuda::memory::Buffer;

struct Args {
    int batch = 32;
    int seq_len = 128;
    int hidden = 512;
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -h, --help        Show this help and exit\n");
    printf("  --batch <n>       Batch size (positive int)\n");
    printf("  --seq_len <n>     Sequence length (positive int)\n");
    printf("  --hidden <n>      Hidden dimension (positive int)\n");
}

// Positive-int argument parse (fail fast instead of silently parsing garbage
// as 0 via atoi — the old example did `atoi(argv[i])` with no bounds check).
bool parse_positive_int(const char* arg, int* out, const char* name) {
    char* end = nullptr;
    const long v = strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || v <= 0 || v > 1 << 24) {
        fprintf(stderr, "Error: --%s expects a positive integer, got '%s'\n",
                name, arg);
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

// Returns 1 on "--help"/"-h" (caller prints usage, exits 0), 0 on success,
// -1 on a parse error (caller prints the error, exits 1).
int parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &args.batch, "batch")) return -1;
        } else if (strcmp(argv[i], "--seq_len") == 0 && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &args.seq_len, "seq_len")) return -1;
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &args.hidden, "hidden")) return -1;
        } else {
            fprintf(stderr, "Error: unknown or incomplete option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    if (args.hidden > (1 << 20)) {
        fprintf(stderr, "Error: --hidden is too large\n");
        return -1;
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    Args args;
    const int parse = parse_args(argc, argv, args);
    if (parse < 0) {
        return 1;    // error already reported
    }
    if (parse > 0) {
        return 0;    // --help printed
    }

    printf("Nova Neural Net Primitives Example\n");
    printf("Batch: %d, SeqLen: %d, Hidden: %d\n",
           args.batch, args.seq_len, args.hidden);

    const int N = args.batch * args.seq_len;
    const int K = args.hidden;
    const int M = args.hidden;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    // Inputs on host, then uploaded (Buffer owns device memory only).
    std::vector<float> h_input(static_cast<size_t>(N) * K);
    std::vector<float> h_weight(static_cast<size_t>(K) * M);
    std::vector<float> h_bias(static_cast<size_t>(M), 0.0f);
    for (float& v : h_input) v = dist(rng);
    for (float& v : h_weight) v = dist(rng);

    Buffer<float> input(static_cast<size_t>(N) * K);
    Buffer<float> weight(static_cast<size_t>(K) * M);
    Buffer<float> bias(static_cast<size_t>(M));
    Buffer<float> output(static_cast<size_t>(N) * M);
    Buffer<float> ln_out(static_cast<size_t>(N) * M);
    input.copy_from(h_input.data(), h_input.size());
    weight.copy_from(h_weight.data(), h_weight.size());
    bias.fill(0.0f);

    // 1+2. Fused matmul + bias + ReLU (the v2.13 fusion layer; the removed
    //      matmul_bias primitive folded into it).
    printf("Running fused matmul + bias + ReLU...\n");
    cuda::neural::fusion::FusedMatmulBiasAct fused(cuda::neural::fusion::MatmulBiasActConfig{
        /*handle=*/nullptr, cuda::neural::fusion::ActivationType::ReLU,
        /*relu_threshold=*/0.0f, /*use_cuda_fusion=*/true,
        /*max_workspace_bytes=*/1 << 20});
    fused.forward(input.data(), weight.data(), bias.data(), output.data(), N, M, K);

    // 3. LayerNorm over the hidden dim (identity affine gamma=1/beta=0).
    //    The mean/variance are outputs we don't need for the demo.
    printf("Applying LayerNorm...\n");
    Buffer<float> gamma(static_cast<size_t>(M));
    Buffer<float> beta(static_cast<size_t>(M));
    Buffer<float> mean(static_cast<size_t>(N));
    Buffer<float> variance(static_cast<size_t>(N));
    {
        std::vector<float> ones(static_cast<size_t>(M), 1.0f);
        std::vector<float> zeros(static_cast<size_t>(M), 0.0f);
        gamma.copy_from(ones.data(), ones.size());
        beta.copy_from(zeros.data(), zeros.size());
    }
    cuda::neural::layer_norm(output.data(), gamma.data(), beta.data(),
                             ln_out.data(), mean.data(), variance.data(),
                             N, M, 1e-5f);

    // 4. Softmax over the hidden (classes) dim, per row.
    printf("Applying softmax...\n");
    cuda::neural::softmax(ln_out.data(), output.data(), N, M);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify: every row is now a probability distribution over hidden classes
    // (the original example's "should be ~batch" row check was wrong — softmax
    // rows sum to 1, not to the batch count).
    std::vector<float> h_out(static_cast<size_t>(N) * M);
    output.copy_to(h_out.data(), h_out.size());
    bool valid = true;
    for (int r = 0; r < std::min(8, N); ++r) {
        double row_sum = 0.0;
        for (int c = 0; c < M; ++c) {
            row_sum += h_out[static_cast<size_t>(r) * M + c];
        }
        printf("  row %d sum: %.4f\n", r, row_sum);
        if (std::abs(row_sum - 1.0) > 1e-3) {
            valid = false;
        }
    }

    if (valid) {
        printf("\nAll checks passed! Neural net forward pass successful.\n");
        return 0;
    }
    fprintf(stderr, "\nWarning: some softmax rows do not sum to 1.\n");
    return 1;
}
