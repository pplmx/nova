#pragma once

#include <cuda_runtime.h>
#include <string>

namespace cuda::neural::fusion {

    struct FusionConfig {
        bool fuse_matmul_bias = true;
        bool fuse_matmul_bias_activation = true;
        bool fuse_layernorm_softmax = true;
        int max_registers_per_thread = 256;
        int min_elements_for_fusion = 64;
    };

    class FusionKernelRegistry {
    public:
        static FusionKernelRegistry& instance();

        void register_fusion(const std::string& name, bool enabled);
        void enable_fusion(const std::string& name);
        void disable_fusion(const std::string& name);
        bool is_fusion_enabled(const std::string& name) const;

        void set_config(const FusionConfig& config);
        FusionConfig get_config() const;

    private:
        FusionKernelRegistry() = default;

        FusionConfig config_;
        bool matmul_bias_enabled_ = true;
        bool matmul_bias_activation_enabled_ = true;
        bool layernorm_softmax_enabled_ = true;
    };

    void fused_matmul_bias(const float* matmul_out, const float* bias, float* output, int batch_size, int num_features, cudaStream_t stream = nullptr);

    void fused_matmul_bias_relu(const float* matmul_out, const float* bias, float* output, int batch_size, int num_features, cudaStream_t stream = nullptr);

    void fused_matmul_bias_sigmoid(const float* matmul_out, const float* bias, float* output, int batch_size, int num_features, cudaStream_t stream = nullptr);

    class FusedLayerNormSoftmax {
    public:
        FusedLayerNormSoftmax(int hidden_size, float eps = 1e-5f);
        ~FusedLayerNormSoftmax();

        void forward(const float* input, float* output, const float* gamma, const float* beta, int batch_size, int seq_len, cudaStream_t stream = nullptr);

    private:
        int hidden_size_;
        float eps_;
    };

    bool should_use_fused_kernel(const std::string& op, int batch_size, int num_features);

}  // namespace cuda::neural::fusion
