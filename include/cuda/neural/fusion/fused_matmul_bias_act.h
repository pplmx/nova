#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <string>
#include <variant>
#include <memory>

namespace cuda::neural::fusion {

enum class ActivationType {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    GELU
};

struct MatmulBiasActConfig {
    cublasHandle_t handle = nullptr;
    ActivationType activation = ActivationType::ReLU;
    float relu_threshold = 0.0f;
    bool use_cuda_fusion = true;
    int max_workspace_bytes = 1024 * 1024 * 1024;  // 1GB workspace
};

class FusedMatmulBiasAct {
public:
    explicit FusedMatmulBiasAct(const MatmulBiasActConfig& config);
    ~FusedMatmulBiasAct();

    void forward(
        const float* A,
        const float* B,
        const float* bias,
        float* C,
        int m,
        int n,
        int k,
        cudaStream_t stream = nullptr
    );

    void set_activation(ActivationType activation);
    ActivationType get_activation() const;

    void enable_cuda_fusion();
    void disable_cuda_fusion();
    bool is_cuda_fusion_enabled() const;

private:
    void forward_fallback(
        const float* A,
        const float* B,
        const float* bias,
        float* C,
        int m,
        int n,
        int k,
        cudaStream_t stream
    );

    MatmulBiasActConfig config_;
    // Private cuBLAS handle used when the caller did not inject one via
    // config_.handle. Never fall back to cuda::neural's global handle: setting
    // a stream on that shared handle permanently rebinds it, and once the
    // caller's stream is destroyed the global handle dangles and every later
    // cublasSgemm via cuda::neural::matmul fails with EXECUTION_FAILED.
    cublasHandle_t owned_handle_ = nullptr;
    cublasLtHandle_t lt_handle_ = nullptr;
    bool cuda_fusion_available_ = false;
    float* workspace_ = nullptr;
};

struct FusionPolicy {
    bool fuse_matmul_bias = true;
    bool fuse_matmul_bias_activation = true;
    bool fuse_layernorm_softmax = true;
    bool use_cuda_fusion_when_available = true;
    int min_elements_for_fusion = 64;
    int max_fusion_depth = 4;
};

class FusionPolicyManager {
public:
    static FusionPolicyManager& instance();

    void set_policy(const FusionPolicy& policy);
    FusionPolicy get_policy() const;

    bool should_fuse(const std::string& op_name, int element_count) const;
    void enable_op(const std::string& op_name);
    void disable_op(const std::string& op_name);

    std::string get_policy_summary() const;

private:
    FusionPolicyManager() = default;
    FusionPolicy policy_;
};

template <typename T>
void apply_activation(
    T* data,
    size_t num_elements,
    ActivationType activation,
    float threshold = 0.0f,
    cudaStream_t stream = nullptr
);

template <typename T>
__global__ void relu_kernel(T* data, size_t num_elements, T threshold);

template <typename T>
__global__ void sigmoid_kernel(T* data, size_t num_elements);

template <typename T>
__global__ void tanh_kernel(T* data, size_t num_elements);

template <typename T>
__global__ void gelu_kernel(T* data, size_t num_elements);

}  // namespace cuda::neural::fusion
