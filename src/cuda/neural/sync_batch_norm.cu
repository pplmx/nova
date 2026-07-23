#include "cuda/neural/sync_batch_norm.h"

#include "cuda/device/reduce_kernels.h"
#include "cuda/distributed/reduce.h"
#include "cuda/mesh/device_mesh.h"

#include "cuda/device/error.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace cuda::neural {

namespace {

template <typename T>
__global__ void compute_mean_kernel(
    const T* input,
    T* mean,
    int batch_size,
    int num_features,
    int spatial_size,
    T inv_n
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    T sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            int idx = (b * num_features + feature) * spatial_size + s;
            sum += input[idx];
        }
    }
    mean[feature] = sum * inv_n;
}

template <typename T>
__global__ void subtract_mean_kernel(
    const T* input,
    const T* mean,
    T* output,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    output[idx] = input[idx] - mean[feature];
}

template <typename T>
__global__ void compute_variance_kernel(
    const T* centered_input,
    T* variance,
    int batch_size,
    int num_features,
    int spatial_size,
    T inv_n
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    T sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            int idx = (b * num_features + feature) * spatial_size + s;
            T val = centered_input[idx];
            sum += val * val;
        }
    }
    variance[feature] = sum * inv_n;
}

template <typename T>
__global__ void normalize_kernel(
    const T* input,
    const T* mean,
    const T* variance,
    T* output,
    int batch_size,
    int num_features,
    int spatial_size,
    T eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    T std = sqrtf(variance[feature] + eps);
    output[idx] = (input[idx] - mean[feature]) / std;
}

template <typename T>
__global__ void scale_bias_kernel(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    output[idx] = input[idx] * gamma[feature] + beta[feature];
}

template <typename T>
__global__ void inference_normalize_kernel(
    const T* input,
    const T* mean,
    const T* var,
    T* output,
    int batch_size,
    int num_features,
    int spatial_size,
    T eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    T std = sqrtf(var[feature] + eps);
    output[idx] = (input[idx] - mean[feature]) / std;
}

__global__ void update_running_stats_kernel(
    float* running_mean,
    const float* batch_mean,
    float* running_var,
    const float* batch_var,
    int num_features,
    float momentum
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    running_mean[feature] = (1.0f - momentum) * running_mean[feature] +
                            momentum * batch_mean[feature];
    running_var[feature] = (1.0f - momentum) * running_var[feature] +
                           momentum * batch_var[feature];
}

template <typename T>
__global__ void backward_dxnorm_kernel(
    const T* d_output,
    const T* gamma,
    T* d_x_norm,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    d_x_norm[idx] = d_output[idx] * gamma[feature];
}

template <typename T>
__global__ void backward_dvar_kernel(
    const T* d_x_norm,
    const T* centered,
    const T* variance,
    T* d_var,
    int batch_size,
    int num_features,
    int spatial_size,
    T inv_n,
    T eps
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    T sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            int idx = (b * num_features + feature) * spatial_size + s;
            sum += d_x_norm[idx] * centered[idx];
        }
    }
    T var_eps = sqrtf(variance[feature] + eps);
    d_var[feature] = sum * (-0.5f) * powf(var_eps, -3.0f);
}

template <typename T>
__global__ void backward_dmean_kernel(
    const T* d_x_norm,
    const T* d_var,
    const T* variance,
    const T* centered,
    T* d_mean,
    int batch_size,
    int num_features,
    int spatial_size,
    T inv_n,
    T eps
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    T sum_dxnorm = 0.0f;
    T sum_centered = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            int idx = (b * num_features + feature) * spatial_size + s;
            sum_dxnorm += d_x_norm[idx];
            sum_centered += centered[idx];
        }
    }
    T inv_var_eps = 1.0f / sqrtf(variance[feature] + eps);
    d_mean[feature] = sum_dxnorm * (-inv_var_eps) +
                      d_var[feature] * (-2.0f * inv_n) * sum_centered;
}

template <typename T>
__global__ void backward_dinput_kernel(
    const T* d_x_norm,
    const T* centered,
    const T* d_var,
    const T* d_mean,
    T* d_input,
    int batch_size,
    int num_features,
    int spatial_size,
    T inv_n,
    T eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    T var_eps = sqrtf(d_var[feature] + eps);

    T dx_norm_term = d_x_norm[idx] / var_eps;
    T dvar_term = d_var[feature] * 2.0f * centered[idx] * inv_n;
    T dmean_term = d_mean[feature] * inv_n;

    d_input[idx] = dx_norm_term + dvar_term + dmean_term;
}

template <typename T>
__global__ void backward_dgamma_kernel(
    const T* d_output,
    const T* normalized,
    T* d_gamma,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    T sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            int idx = (b * num_features + feature) * spatial_size + s;
            sum += d_output[idx] * normalized[idx];
        }
    }
    d_gamma[feature] = sum;
}

template <typename T>
__global__ void backward_dbeta_kernel(
    const T* d_output,
    T* d_beta,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    T sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            int idx = (b * num_features + feature) * spatial_size + s;
            sum += d_output[idx];
        }
    }
    d_beta[feature] = sum;
}

template <typename T>
__global__ void compute_centered_kernel(
    const T* input,
    const T* mean,
    T* centered,
    int batch_size,
    int num_features,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    centered[idx] = input[idx] - mean[feature];
}

template <typename T>
__global__ void compute_normalized_kernel(
    const T* centered,
    const T* variance,
    T* normalized,
    int batch_size,
    int num_features,
    int spatial_size,
    T eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = batch_size * num_features * spatial_size;
    if (idx >= n) return;

    int feature = (idx / spatial_size) % num_features;
    T std = sqrtf(variance[feature] + eps);
    normalized[idx] = centered[idx] / std;
}

} // anonymous namespace

SyncBatchNorm::SyncBatchNorm(int num_features, float eps, float momentum)
    : num_features_(num_features),
      eps_(eps),
      momentum_(momentum),
      training_(true),
      initialized_(false) {

    CUDA_CHECK(cudaMalloc(&running_mean_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&running_var_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gamma_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&beta_, num_features * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&saved_mean_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&saved_var_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&saved_input_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&saved_output_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&normalized_, num_features * sizeof(float)));

    float* h_data = new float[num_features];
    for (int i = 0; i < num_features; ++i) h_data[i] = 0.0f;
    CUDA_CHECK(cudaMemcpy(running_mean_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(running_var_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(beta_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_features; ++i) h_data[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(gamma_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_data;
}

SyncBatchNorm::~SyncBatchNorm() {
    cudaFree(running_mean_);
    cudaFree(running_var_);
    cudaFree(gamma_);
    cudaFree(beta_);
    cudaFree(saved_mean_);
    cudaFree(saved_var_);
    cudaFree(saved_input_);
    cudaFree(saved_output_);
    cudaFree(normalized_);
}

void SyncBatchNorm::set_training(bool training) {
    training_ = training;
}

void SyncBatchNorm::forward_training(
    const float* input,
    float* output,
    int batch_size,
    int spatial_size,
    cudaStream_t stream
) {
    int n = batch_size * num_features_ * spatial_size;
    float inv_n = 1.0f / static_cast<float>(batch_size * spatial_size);

    int block_size = 256;
    int grid_size = (num_features_ + block_size - 1) / block_size;

    compute_mean_kernel<float><<<grid_size, block_size, 0, stream>>>(
        input, saved_mean_, batch_size, num_features_, spatial_size, inv_n);

    auto& mesh = mesh::DeviceMesh::instance();
    int device_count = mesh.device_count();

    if (device_count > 1) {
        distributed::DistributedReduce::all_reduce_async(
            saved_mean_, saved_mean_, num_features_,
            distributed::ReductionOp::Sum,
            stream
        );
    }

    subtract_mean_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        input, saved_mean_, output, batch_size, num_features_, spatial_size);

    compute_variance_kernel<float><<<grid_size, block_size, 0, stream>>>(
        output, saved_var_, batch_size, num_features_, spatial_size, inv_n);

    if (device_count > 1) {
        distributed::DistributedReduce::all_reduce_async(
            saved_var_, saved_var_, num_features_,
            distributed::ReductionOp::Sum,
            stream
        );
    }

    normalize_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        output, saved_mean_, saved_var_, output,
        batch_size, num_features_, spatial_size, eps_);

    scale_bias_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        output, gamma_, beta_, output,
        batch_size, num_features_, spatial_size);

    update_running_stats_kernel<<<grid_size, block_size, 0, stream>>>(
        running_mean_, saved_mean_,
        running_var_, saved_var_,
        num_features_, momentum_);

    initialized_ = true;
}

void SyncBatchNorm::forward_inference(
    const float* input,
    float* output,
    int batch_size,
    int spatial_size,
    cudaStream_t stream
) {
    int n = batch_size * num_features_ * spatial_size;
    int block_size = 256;

    inference_normalize_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        input, running_mean_, running_var_, output,
        batch_size, num_features_, spatial_size, eps_);

    scale_bias_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        output, gamma_, beta_, output,
        batch_size, num_features_, spatial_size);
}

void SyncBatchNorm::backward(
    const float* input,
    const float* d_output,
    float* d_input,
    float* d_gamma,
    float* d_beta,
    int batch_size,
    int spatial_size,
    cudaStream_t stream
) {
    int n = batch_size * num_features_ * spatial_size;
    int block_size = 256;
    int grid_size = (num_features_ + block_size - 1) / block_size;
    float inv_n = 1.0f / static_cast<float>(batch_size * spatial_size);

    float* d_x_norm;
    float* centered_input;
    float* normalized_tmp;
    float* d_var;
    float* d_mean;
    CUDA_CHECK(cudaMalloc(&d_x_norm, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&centered_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&normalized_tmp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, num_features_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, num_features_ * sizeof(float)));

    compute_centered_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        input, saved_mean_, centered_input,
        batch_size, num_features_, spatial_size);

    compute_normalized_kernel<float><<<grid_size, block_size, 0, stream>>>(
        centered_input, saved_var_, normalized_tmp,
        batch_size, num_features_, spatial_size, eps_);

    backward_dxnorm_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        d_output, gamma_, d_x_norm,
        batch_size, num_features_, spatial_size);

    backward_dvar_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_x_norm, centered_input, saved_var_, d_var,
        batch_size, num_features_, spatial_size, inv_n, eps_);

    backward_dmean_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_x_norm, d_var, saved_var_, centered_input, d_mean,
        batch_size, num_features_, spatial_size, inv_n, eps_);

    backward_dinput_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        d_x_norm, centered_input, d_var, d_mean, d_input,
        batch_size, num_features_, spatial_size, inv_n, eps_);

    backward_dgamma_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_output, normalized_tmp, d_gamma,
        batch_size, num_features_, spatial_size);

    backward_dbeta_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_output, d_beta,
        batch_size, num_features_, spatial_size);

    auto& mesh = mesh::DeviceMesh::instance();
    if (mesh.device_count() > 1) {
        distributed::DistributedReduce::all_reduce_async(
            d_gamma, d_gamma, num_features_,
            distributed::ReductionOp::Sum,
            stream
        );
        distributed::DistributedReduce::all_reduce_async(
            d_beta, d_beta, num_features_,
            distributed::ReductionOp::Sum,
            stream
        );
    }

    CUDA_CHECK(cudaFree(d_x_norm));
    CUDA_CHECK(cudaFree(centered_input));
    CUDA_CHECK(cudaFree(normalized_tmp));
    CUDA_CHECK(cudaFree(d_var));
    CUDA_CHECK(cudaFree(d_mean));
}

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
    cudaStream_t stream
) {
    SyncBatchNorm bn(num_features, eps, momentum);
    bn.mutable_gamma();
    CUDA_CHECK(cudaMemcpy(bn.mutable_gamma(), gamma, num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bn.mutable_beta(), beta, num_features * sizeof(float), cudaMemcpyHostToDevice));

    bn.forward_training(input, output, batch_size, spatial_size, stream);
}

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
    cudaStream_t stream
) {
    int n = batch_size * num_features * spatial_size;
    int block_size = 256;

    inference_normalize_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        input, running_mean, running_var, output,
        batch_size, num_features, spatial_size, eps);

    scale_bias_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        output, gamma, beta, output,
        batch_size, num_features, spatial_size);
}

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
    cudaStream_t stream
) {
    SyncBatchNorm bn(num_features, eps);
    CUDA_CHECK(cudaMemcpy(bn.mutable_saved_mean(), saved_mean, num_features * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(bn.mutable_saved_var(), saved_var, num_features * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(bn.mutable_gamma(), gamma, num_features * sizeof(float), cudaMemcpyDeviceToDevice));
    bn.backward(input, d_output, d_input, d_gamma, d_beta, batch_size, spatial_size, stream);
}

} // namespace cuda::neural
