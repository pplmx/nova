#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <cstddef>
#include <cstdint>
#include "cuda/device/device_utils.h"

struct ImageDimensions {
    size_t width;
    size_t height;
    size_t channels;
};

enum class PixelFormat { UCHAR3, FLOAT3 };

template<PixelFormat Format>
class ImageBuffer;

template<>
class ImageBuffer<PixelFormat::UCHAR3> {
public:
    using PixelType = uint8_t;

private:
    PixelType* d_data_;
    ImageDimensions dims_;
    struct Deleter {
        void operator()(PixelType* p) const {
            cudaFree(p);
        }
    };
    std::unique_ptr<PixelType, Deleter> data_owner_;

public:
    ImageBuffer() : d_data_(nullptr), dims_{0, 0, 0}, data_owner_(nullptr) {}

    explicit ImageBuffer(size_t width, size_t height, size_t channels = 3)
        : dims_{width, height, channels} {
        if (size() > 0) {
            PixelType* raw_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&raw_ptr, size() * sizeof(PixelType)));
            data_owner_.reset(raw_ptr);
            d_data_ = data_owner_.get();
        }
    }

    [[nodiscard]] PixelType* data() const { return d_data_; }
    [[nodiscard]] ImageDimensions dimensions() const { return dims_; }
    [[nodiscard]] size_t size() const { return dims_.width * dims_.height * dims_.channels; }
    [[nodiscard]] explicit operator bool() const { return d_data_ != nullptr; }

    void upload(const PixelType* h_data) {
        if (d_data_ && h_data) {
            CUDA_CHECK(cudaMemcpy(d_data_, h_data, size() * sizeof(PixelType), cudaMemcpyHostToDevice));
        }
    }

    void download(PixelType* h_data) const {
        if (d_data_ && h_data) {
            CUDA_CHECK(cudaMemcpy(h_data, d_data_, size() * sizeof(PixelType), cudaMemcpyDeviceToHost));
        }
    }
};

template<>
class ImageBuffer<PixelFormat::FLOAT3> {
public:
    using PixelType = float;

private:
    PixelType* d_data_;
    ImageDimensions dims_;
    struct Deleter {
        void operator()(PixelType* p) const {
            cudaFree(p);
        }
    };
    std::unique_ptr<PixelType, Deleter> data_owner_;

public:
    ImageBuffer() : d_data_(nullptr), dims_{0, 0, 0}, data_owner_(nullptr) {}

    explicit ImageBuffer(size_t width, size_t height, size_t channels = 3)
        : dims_{width, height, channels} {
        if (size() > 0) {
            PixelType* raw_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&raw_ptr, size() * sizeof(PixelType)));
            data_owner_.reset(raw_ptr);
            d_data_ = data_owner_.get();
        }
    }

    [[nodiscard]] PixelType* data() const { return d_data_; }
    [[nodiscard]] ImageDimensions dimensions() const { return dims_; }
    [[nodiscard]] size_t size() const { return dims_.width * dims_.height * dims_.channels; }
    [[nodiscard]] explicit operator bool() const { return d_data_ != nullptr; }

    void upload(const PixelType* h_data) {
        if (d_data_ && h_data) {
            CUDA_CHECK(cudaMemcpy(d_data_, h_data, size() * sizeof(PixelType), cudaMemcpyHostToDevice));
        }
    }

    void download(PixelType* h_data) const {
        if (d_data_ && h_data) {
            CUDA_CHECK(cudaMemcpy(h_data, d_data_, size() * sizeof(PixelType), cudaMemcpyDeviceToHost));
        }
    }
};
