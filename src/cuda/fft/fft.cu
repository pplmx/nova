#include "cuda/fft/fft.h"

#include "cuda/device/error.h"

namespace {

#define CUFFT_CHECK(call)                                                     \
    do {                                                                      \
        cufftResult err = call;                                                \
        if (err != CUFFT_SUCCESS) {                                           \
            throw std::runtime_error("cuFFT error: " + std::to_string(err));  \
        }                                                                     \
    } while (0)

}  // anonymous namespace

namespace cuda::fft {

FFTPlan::FFTPlan(size_t size, Direction direction, TransformType type)
    : size_(size),
      nx_(size),
      ny_(1),
      nz_(1),
      direction_(direction),
      type_(type) {
    create_plan_1d(size, type);
}

FFTPlan::FFTPlan(size_t nx, size_t ny, Direction direction, TransformType type)
    : size_(nx * ny),
      nx_(nx),
      ny_(ny),
      nz_(1),
      direction_(direction),
      type_(type) {
    create_plan_2d(nx, ny, type);
}

FFTPlan::FFTPlan(size_t nx, size_t ny, size_t nz, Direction direction, TransformType type)
    : size_(nx * ny * nz),
      nx_(nx),
      ny_(ny),
      nz_(nz),
      direction_(direction),
      type_(type) {
    create_plan_3d(nx, ny, nz, type);
}

void FFTPlan::create_plan_1d(size_t size, TransformType type) {
    int n = static_cast<int>(size);
    cufftType cufft_type;

    switch (type) {
        case TransformType::RealToComplex:
            cufft_type = CUFFT_R2C;
            break;
        case TransformType::DoubleRealToComplex:
            cufft_type = CUFFT_D2Z;
            break;
        case TransformType::ComplexToReal:
            cufft_type = CUFFT_C2R;
            break;
        case TransformType::ComplexToComplex:
            cufft_type = CUFFT_C2C;
            break;
        default:
            throw std::invalid_argument("Unsupported transform type");
    }

    CUFFT_CHECK(cufftPlan1d(&plan_, n, cufft_type, 1));
}

void FFTPlan::create_plan_2d(size_t nx, size_t ny, TransformType type) {
    cufftType cufft_type;

    switch (type) {
        case TransformType::RealToComplex:
            cufft_type = CUFFT_R2C;
            break;
        case TransformType::DoubleRealToComplex:
            cufft_type = CUFFT_D2Z;
            break;
        default:
            throw std::invalid_argument("Unsupported transform type for 2D");
    }

    CUFFT_CHECK(cufftPlan2d(&plan_, nx, ny, cufft_type));
}

void FFTPlan::create_plan_3d(size_t nx, size_t ny, size_t nz, TransformType type) {
    cufftType cufft_type;

    switch (type) {
        case TransformType::RealToComplex:
            cufft_type = CUFFT_R2C;
            break;
        case TransformType::DoubleRealToComplex:
            cufft_type = CUFFT_D2Z;
            break;
        default:
            throw std::invalid_argument("Unsupported transform type for 3D");
    }

    CUFFT_CHECK(cufftPlan3d(&plan_, nx, ny, nz, cufft_type));
}

void FFTPlan::destroy_plan() {
    if (plan_ != 0 && owns_plan_) {
        cufftDestroy(plan_);
        plan_ = 0;
    }
}

void FFTPlan::forward(const float* input, cuComplex* output, cudaStream_t stream) {
    if (stream) {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
    }
    CUFFT_CHECK(cufftExecR2C(plan_, const_cast<float*>(input), output));
}

void FFTPlan::forward(const double* input, cuDoubleComplex* output, cudaStream_t stream) {
    if (stream) {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
    }
    CUFFT_CHECK(cufftExecD2Z(plan_, const_cast<double*>(input), output));
}

void FFTPlan::inverse(const cuComplex* input, float* output, cudaStream_t stream) {
    if (stream) {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
    }
    CUFFT_CHECK(cufftExecC2R(plan_, const_cast<cuComplex*>(input), output));
}

void FFTPlan::inverse(const cuDoubleComplex* input, double* output, cudaStream_t stream) {
    if (stream) {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
    }
    CUFFT_CHECK(cufftExecZ2D(plan_, const_cast<cuDoubleComplex*>(input), output));
}

void FFTPlan::transform(Direction direction, const cuComplex* input,
                       cuComplex* output, cudaStream_t stream) {
    if (stream) {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
    }
    auto dir = (direction == Direction::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
    CUFFT_CHECK(cufftExecC2C(plan_, const_cast<cuComplex*>(input), output, dir));
}

void FFTPlan::transform(Direction direction, const cuDoubleComplex* input,
                       cuDoubleComplex* output, cudaStream_t stream) {
    if (stream) {
        CUFFT_CHECK(cufftSetStream(plan_, stream));
    }
    auto dir = (direction == Direction::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
    CUFFT_CHECK(cufftExecZ2Z(plan_, const_cast<cuDoubleComplex*>(input), output, dir));
}

void forward_inplace(FFTPlan& plan, cuComplex* data, cudaStream_t stream) {
    plan.transform(Direction::Forward, data, data, stream);
}

void inverse_inplace(FFTPlan& plan, cuComplex* data, cudaStream_t stream) {
    plan.transform(Direction::Inverse, data, data, stream);
}

__global__ void magnitude_kernel(const cuComplex* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = hypotf(input[idx].x, input[idx].y);
    }
}

__global__ void power_kernel(const cuComplex* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx].x * input[idx].x + input[idx].y * input[idx].y;
    }
}

__global__ void scale_kernel(float* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

void magnitude(const cuComplex* input, float* output, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    magnitude_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
}

void power_spectrum(const cuComplex* input, float* output, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    power_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
}

template FFTPlan make_fft_plan<float>(size_t size, Direction dir);
template FFTPlan make_fft_plan<double>(size_t size, Direction dir);

template <typename T>
FFTPlan make_fft_plan(size_t size, Direction dir) {
    return FFTPlan(size, dir, TransformType::RealToComplex);
}

}  // namespace cuda::fft
