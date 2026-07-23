#pragma once

#include <cuda_runtime.h>

namespace cuda::neural {

struct SyncBatchNormParams {
    int num_features;
    float eps = 1e-5f;
    float momentum = 0.1f;
    bool affine = true;
};

class SyncBatchNorm {
public:
    SyncBatchNorm(int num_features, float eps = 1e-5f, float momentum = 0.1f);
    ~SyncBatchNorm();

    SyncBatchNorm(const SyncBatchNorm&) = delete;
    SyncBatchNorm& operator=(const SyncBatchNorm&) = delete;
    SyncBatchNorm(SyncBatchNorm&&) = default;
    SyncBatchNorm& operator=(SyncBatchNorm&&) = default;

    void set_training(bool training);
    bool is_training() const { return training_; }

    void forward_training(
        const float* input,
        float* output,
        int batch_size,
        int spatial_size,
        cudaStream_t stream = nullptr
    );

    void forward_inference(
        const float* input,
        float* output,
        int batch_size,
        int spatial_size,
        cudaStream_t stream = nullptr
    );

    void backward(
        const float* input,
        const float* d_output,
        float* d_input,
        float* d_gamma,
        float* d_beta,
        int batch_size,
        int spatial_size,
        cudaStream_t stream = nullptr
    );

    const float* running_mean() const { return running_mean_; }
    const float* running_var() const { return running_var_; }
    const float* gamma() const { return gamma_; }
    const float* beta() const { return beta_; }

    float* mutable_running_mean() { return running_mean_; }
    float* mutable_running_var() { return running_var_; }
    float* mutable_gamma() { return gamma_; }
    float* mutable_beta() { return beta_; }
    float* mutable_saved_mean() { return saved_mean_; }
    float* mutable_saved_var() { return saved_var_; }

    int num_features() const { return num_features_; }
    float eps() const { return eps_; }
    float momentum() const { return momentum_; }

private:
    int num_features_;
    float eps_;
    float momentum_;
    bool training_;

    float* running_mean_;
    float* running_var_;
    float* gamma_;
    float* beta_;

    float* saved_mean_;
    float* saved_var_;
    float* saved_input_;
    float* normalized_;
    float* saved_output_;

    bool initialized_;
};

void sync_batch_norm_forward_training(
    const float* input,
    float* output,
    float* saved_mean,
    float* saved_var,
    float* gamma,
    float* beta,
    float* running_mean,
    float* running_var,
    int batch_size,
    int num_features,
    int spatial_size,
    float eps,
    float momentum,
    cudaStream_t stream = nullptr
);

void sync_batch_norm_forward_inference(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int batch_size,
    int num_features,
    int spatial_size,
    float eps,
    cudaStream_t stream = nullptr
);

void sync_batch_norm_backward(
    const float* input,
    const float* d_output,
    float* d_input,
    float* d_gamma,
    float* d_beta,
    const float* saved_mean,
    const float* saved_var,
    const float* gamma,
    int batch_size,
    int num_features,
    int spatial_size,
    float eps,
    cudaStream_t stream = nullptr
);

} // namespace cuda::neural
