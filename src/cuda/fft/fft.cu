#include "cuda/fft/fft.h"

#include <limits>

#include "cuda/device/error.h"

namespace {

#define CUFFT_CHECK(call)                                                                                                                                \
    do {                                                                                                                                                 \
        cufftResult err = call;                                                                                                                          \
        if (err != CUFFT_SUCCESS) {                                                                                                                      \
            throw std::runtime_error(std::string("cuFFT error: ") + ::cuda::fft::result_name(err) + " (" + std::to_string(static_cast<int>(err)) + ")"); \
        }                                                                                                                                                \
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
        // A zero-length transform has no meaning, and a size above INT_MAX
        // truncates to an implementation-defined int — the plan was then created for
        // a completely different length (or rejected by cuFFT for a reason that
        // named neither the parameter nor the cause). Validate before narrowing.
        if (size == 0) {
            throw std::invalid_argument("FFTPlan: transform size must be positive, got 0");
        }
        if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FFTPlan: transform size " + std::to_string(size) + " exceeds the cuFFT limit of " + std::to_string(std::numeric_limits<int>::max()));
        }
        const int n = static_cast<int>(size);
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
        if (nx == 0 || ny == 0) {
            throw std::invalid_argument("FFTPlan: 2D dimensions must be positive, got " + std::to_string(nx) + " x " + std::to_string(ny));
        }
        if (nx > static_cast<size_t>(std::numeric_limits<int>::max()) || ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FFTPlan: 2D dimensions exceed the cuFFT int limit, got " + std::to_string(nx) + " x " + std::to_string(ny));
        }
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
                throw std::invalid_argument("Unsupported transform type for 2D");
        }

        CUFFT_CHECK(cufftPlan2d(&plan_, nx, ny, cufft_type));
    }

    void FFTPlan::create_plan_3d(size_t nx, size_t ny, size_t nz, TransformType type) {
        if (nx == 0 || ny == 0 || nz == 0) {
            throw std::invalid_argument("FFTPlan: 3D dimensions must be positive, got " + std::to_string(nx) + " x " + std::to_string(ny) + " x " + std::to_string(nz));
        }
        if (nx > static_cast<size_t>(std::numeric_limits<int>::max()) || ny > static_cast<size_t>(std::numeric_limits<int>::max())
            || nz > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FFTPlan: 3D dimensions exceed the cuFFT int limit, got " + std::to_string(nx) + " x " + std::to_string(ny) + " x " + std::to_string(nz));
        }
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

    void FFTPlan::transform(Direction direction, const cuComplex* input, cuComplex* output, cudaStream_t stream) {
        if (stream) {
            CUFFT_CHECK(cufftSetStream(plan_, stream));
        }
        auto dir = (direction == Direction::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
        CUFFT_CHECK(cufftExecC2C(plan_, const_cast<cuComplex*>(input), output, dir));
    }

    void FFTPlan::transform(Direction direction, const cuDoubleComplex* input, cuDoubleComplex* output, cudaStream_t stream) {
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
        // Zero bins is an ordinary request and would make a zero-sized grid; the FFTPlan
        // entry points above already reject n == 0. Both of these also lacked the launch
        // check every other entry point in this file performs, so a failure here would have
        // surfaced at whatever CUDA call ran next.
        if (n == 0) {
            return;
        }
        int block_size = 256;
        int grid_size = static_cast<int>((n + block_size - 1) / block_size);
        magnitude_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void power_spectrum(const cuComplex* input, float* output, size_t n, cudaStream_t stream) {
        if (n == 0) {
            return;
        }
        int block_size = 256;
        int grid_size = static_cast<int>((n + block_size - 1) / block_size);
        power_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    template FFTPlan make_fft_plan<float>(size_t size, Direction dir);
    template FFTPlan make_fft_plan<double>(size_t size, Direction dir);

    template <typename T>
    FFTPlan make_fft_plan(size_t size, Direction dir) {
        return FFTPlan(size, dir, TransformType::RealToComplex);
    }

}  // namespace cuda::fft
