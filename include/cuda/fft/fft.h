#pragma once

/**
 * @file fft.h
 * @brief FFT plan creation and execution API
 */

#include <cufft.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "cuda/device/error.h"
#include "cuda/fft/fft_types.h"

namespace cuda::fft {

class FFTPlan {
public:
    explicit FFTPlan(size_t size, Direction direction = Direction::Forward,
                    TransformType type = TransformType::RealToComplex);

    FFTPlan(size_t nx, size_t ny, Direction direction = Direction::Forward,
            TransformType type = TransformType::RealToComplex);

    FFTPlan(size_t nx, size_t ny, size_t nz, Direction direction = Direction::Forward,
            TransformType type = TransformType::RealToComplex);

    ~FFTPlan() { destroy_plan(); }

    FFTPlan(const FFTPlan&) = delete;
    FFTPlan& operator=(const FFTPlan&) = delete;

    FFTPlan(FFTPlan&& other) noexcept
        : plan_(other.plan_),
          size_(other.size_),
          nx_(other.nx_),
          ny_(other.ny_),
          nz_(other.nz_),
          direction_(other.direction_),
          type_(other.type_),
          owns_plan_(other.owns_plan_) {
        other.plan_ = 0;
        other.owns_plan_ = false;
    }

    FFTPlan& operator=(FFTPlan&& other) noexcept {
        if (this != &other) {
            destroy_plan();
            plan_ = other.plan_;
            size_ = other.size_;
            nx_ = other.nx_;
            ny_ = other.ny_;
            nz_ = other.nz_;
            direction_ = other.direction_;
            type_ = other.type_;
            owns_plan_ = other.owns_plan_;
            other.plan_ = 0;
            other.owns_plan_ = false;
        }
        return *this;
    }

    void forward(const float* input, cuComplex* output, cudaStream_t stream = nullptr);
    void forward(const double* input, cuDoubleComplex* output, cudaStream_t stream = nullptr);

    void inverse(const cuComplex* input, float* output, cudaStream_t stream = nullptr);
    void inverse(const cuDoubleComplex* input, double* output, cudaStream_t stream = nullptr);

    void transform(Direction direction, const cuComplex* input,
                   cuComplex* output, cudaStream_t stream = nullptr);
    void transform(Direction direction, const cuDoubleComplex* input,
                   cuDoubleComplex* output, cudaStream_t stream = nullptr);

    size_t size() const { return size_; }
    Direction direction() const { return direction_; }
    TransformType type() const { return type_; }

    explicit operator bool() const { return plan_ != 0; }
    cufftHandle handle() const { return plan_; }

private:
    cufftHandle plan_ = 0;
    size_t size_ = 0;
    size_t nx_ = 0;
    size_t ny_ = 0;
    size_t nz_ = 0;
    Direction direction_ = Direction::Forward;
    TransformType type_ = TransformType::RealToComplex;
    bool owns_plan_ = true;

    void create_plan_1d(size_t size, TransformType type);
    void create_plan_2d(size_t nx, size_t ny, TransformType type);
    void create_plan_3d(size_t nx, size_t ny, size_t nz, TransformType type);
    void destroy_plan();
};

template <typename T>
[[nodiscard]] FFTPlan make_fft_plan(size_t size, Direction dir = Direction::Forward);

void forward_inplace(FFTPlan& plan, cuComplex* data, cudaStream_t stream = nullptr);
void inverse_inplace(FFTPlan& plan, cuComplex* data, cudaStream_t stream = nullptr);

__global__ void magnitude_kernel(const cuComplex* input, float* output, size_t n);
__global__ void power_kernel(const cuComplex* input, float* output, size_t n);
__global__ void scale_kernel(float* data, size_t n, float scale);

void magnitude(const cuComplex* input, float* output, size_t n, cudaStream_t stream = nullptr);
void power_spectrum(const cuComplex* input, float* output, size_t n, cudaStream_t stream = nullptr);

}  // namespace cuda::fft
