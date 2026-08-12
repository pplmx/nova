#include "cuda/neural/sync_batch_norm.h"

#include "cuda/device/reduce_kernels.h"
#include "cuda/distributed/reduce.h"
#include "cuda/memory/unique_ptr.h"
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

// Scales one per-feature stats buffer in place. Used after the multi-GPU
// all-reduce of per-rank local statistics: the all-reduce sums R local means /
// variances, and the true global statistics are the sum divided by R (the rank
// count), re-derived from the rank's own shard size.
template <typename T>
__global__ void scale_stats_kernel(T* stats, int num_features, T scale) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;
    stats[feature] *= scale;
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
    // `input` is already mean-centered (subtract_mean_kernel ran first), so
    // normalize by dividing by the standard deviation only — subtracting the
    // mean again here produced (x - 2*mean)/std (task-v17c discovery).
    output[idx] = input[idx] / std;
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
__global__ void backward_dxhat_sum_kernel(
    const T* dxhat,
    T* out,
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
            sum += dxhat[idx];
        }
    }
    out[feature] = sum;
}

template <typename T>
__global__ void backward_dmean_global_kernel(
    const T* dxhat_sum,
    const T* variance,
    T* d_mean,
    int num_features,
    T eps
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= num_features) return;

    T inv_var_eps = 1.0f / sqrtf(variance[feature] + eps);
    // d_mean = -sigma^{-1} * sum_dxhat; the sum(x - mean) term vanishes over the
    // full batch (see caller comment).
    d_mean[feature] = dxhat_sum[feature] * (-inv_var_eps);
}

template <typename T>
__global__ void backward_dinput_kernel(
    const T* d_x_norm,
    const T* centered,
    const T* variance,
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
    // Denominator must be the standard deviation of the batch (sqrt of the
    // batch variance), NOT d_var — the variance gradient can be negative, so
    // sqrtf(d_var + eps) produced NaN for every element.
    T inv_var_eps = 1.0f / sqrtf(variance[feature] + eps);

    T dx_norm_term = d_x_norm[idx] * inv_var_eps;
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
      initialized_(false),
      running_mean_(nullptr),
      running_var_(nullptr),
      gamma_(nullptr),
      beta_(nullptr),
      saved_mean_(nullptr),
      saved_var_(nullptr),
      saved_input_(nullptr),
      normalized_(nullptr),
      saved_output_(nullptr) {

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
    try {
        for (int i = 0; i < num_features; ++i) h_data[i] = 0.0f;
        CUDA_CHECK(cudaMemcpy(running_mean_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(running_var_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(beta_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));

        for (int i = 0; i < num_features; ++i) h_data[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(gamma_, h_data, num_features * sizeof(float), cudaMemcpyHostToDevice));
    } catch (...) {
        delete[] h_data;
        throw;
    }
    delete[] h_data;
}

SyncBatchNorm::~SyncBatchNorm() {
    // Not using CUDA_CHECK here: throwing in a destructor is undefined
    // behavior (C++11 destructors default to noexcept).
    if (running_mean_) cudaFree(running_mean_);
    if (running_var_) cudaFree(running_var_);
    if (gamma_) cudaFree(gamma_);
    if (beta_) cudaFree(beta_);
    if (saved_mean_) cudaFree(saved_mean_);
    if (saved_var_) cudaFree(saved_var_);
    if (saved_input_) cudaFree(saved_input_);
    if (saved_output_) cudaFree(saved_output_);
    if (normalized_) cudaFree(normalized_);
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
        // The all-reduce summed R per-rank local means; the global mean is that
        // sum divided by R. Without this the normalize step uses R x the true
        // mean and the multi-GPU output diverges from a single-rank global
        // batch (task-v17c-syncbn-multigpu-backward).
        //
        // Constraint: dividing by the rank count is exact only when every rank
        // passes an identically-sized batch shard (the data-parallel
        // convention). Unequal shards would need a sample-count-weighted
        // aggregation; callers must keep shards equal.
        scale_stats_kernel<float><<<grid_size, block_size, 0, stream>>>(
            saved_mean_, num_features_, 1.0f / static_cast<float>(device_count));
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
        // Same 1/R scaling for the variance: each rank's local variance is
        // computed around the (now-global) mean, the all-reduce sums them, and
        // dividing by R yields the global variance.
        scale_stats_kernel<float><<<grid_size, block_size, 0, stream>>>(
            saved_var_, num_features_, 1.0f / static_cast<float>(device_count));
    }

    normalize_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        output, saved_var_, output,
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

    // On multi-GPU, d_input/d_mean/d_var gradients are normalized by the GLOBAL
    // batch size (all ranks concatenated); each rank accumulates only its local
    // shard, so fold 1/num_ranks into inv_n to match a single-rank global batch
    // (task-v17c-syncbn-multigpu-backward). d_gamma/d_beta are bare sums and are
    // all-reduced below independently of inv_n.
    int device_count = mesh::DeviceMesh::instance().device_count();
    if (device_count > 1) {
        inv_n /= static_cast<float>(device_count);
    }

    // RAII device buffers: the previous raw cudaMalloc/cudaFree pair leaked all
    // already-allocated buffers if any intermediate CUDA_CHECK threw.
    cuda::memory::unique_ptr<float> d_x_norm(n);
    cuda::memory::unique_ptr<float> centered_input(n);
    cuda::memory::unique_ptr<float> normalized_tmp(n);
    cuda::memory::unique_ptr<float> d_var(num_features_);
    cuda::memory::unique_ptr<float> d_mean(num_features_);
    cuda::memory::unique_ptr<float> d_xhat_sum(num_features_);

    compute_centered_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        input, saved_mean_, centered_input.get(),
        batch_size, num_features_, spatial_size);

    compute_normalized_kernel<float><<<grid_size, block_size, 0, stream>>>(
        centered_input.get(), saved_var_, normalized_tmp.get(),
        batch_size, num_features_, spatial_size, eps_);

    backward_dxnorm_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        d_output, gamma_, d_x_norm.get(),
        batch_size, num_features_, spatial_size);

    backward_dvar_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_x_norm.get(), centered_input.get(), saved_var_, d_var.get(),
        batch_size, num_features_, spatial_size, inv_n, eps_);

    // Per-feature sum of dxhat over this rank's shard; all-reduced below so
    // d_mean uses the GLOBAL sum (d_mean must be identical on every rank).
    backward_dxhat_sum_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_x_norm.get(), d_xhat_sum.get(),
        batch_size, num_features_, spatial_size);

    // The mean/variance gradients are global quantities: every rank feeds them
    // into the same d_input formula, so all-reduce the per-feature d_var and
    // d_xhat_sum before assembling d_mean. (d_gamma/d_beta are all-reduced
    // below.) On a single GPU these reduce to identity.
    if (device_count > 1) {
        distributed::DistributedReduce::all_reduce_async(
            d_var.get(), d_var.get(), num_features_,
            distributed::ReductionOp::Sum,
            stream
        );
        distributed::DistributedReduce::all_reduce_async(
            d_xhat_sum.get(), d_xhat_sum.get(), num_features_,
            distributed::ReductionOp::Sum,
            stream
        );
    }

    // Global mean gradient: d_mean = -sigma^{-1} * sum_dxhat. The standard BN
    // derivation also has a term d_var * (-2/N) * sum(x - mean), which is zero
    // over the full batch (sum(x - mean) == 0 by definition of the mean), so it
    // is omitted; including it would require a further all-reduce of sum(x-mean)
    // that is identically zero.
    backward_dmean_global_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_xhat_sum.get(), saved_var_, d_mean.get(), num_features_, eps_);

    backward_dinput_kernel<float><<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        d_x_norm.get(), centered_input.get(), saved_var_, d_var.get(), d_mean.get(), d_input,
        batch_size, num_features_, spatial_size, inv_n, eps_);

    backward_dgamma_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_output, normalized_tmp.get(), d_gamma,
        batch_size, num_features_, spatial_size);

    backward_dbeta_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_output, d_beta,
        batch_size, num_features_, spatial_size);

    if (device_count > 1) {
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
