# Stage 2: CUDA Image Filters - Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement task-by-task.

**Goal:** 实现三个 CUDA 图像滤波器(高斯模糊、Sobel边缘检测、亮度/对比度)及 GoogleTest 测试

**Architecture:** 分离的 header/implementation 文件结构,RAII ImageBuffer,模板支持多像素格式

**Tech Stack:** CUDA C++, GoogleTest, stb_image.h

---

## Task 1: 项目基础设施

**Files:**
- Modify: `CMakeLists.txt`
- Create: `tests/CMakeLists.txt`
- Create: `data/test_patterns.cuh`
- Create: `data/stb_image.h`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p tests data
```

- [ ] **Step 2: 添加 GoogleTest 到 CMakeLists.txt**

在主 `CMakeLists.txt` 中添加 FetchContent 或 submodule 支持:

```cmake
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)
```

- [ ] **Step 3: 创建 tests/CMakeLists.txt**

```cmake
find_package(CUDAToolkit REQUIRED)

add_executable(cu-tests
    image_utils_test.cu
    gaussian_blur_test.cu
    sobel_edge_test.cu
    brightness_test.cu
)

target_link_libraries(cu-tests
    PRIVATE
    GTest::gtest_main
    GTest::gmock
    CUDA::cudart
)

target_include_directories(cu-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data
)

target_compile_options(cu-tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -lineinfo
    >
)

include(GoogleTest)
gtest_discover_tests(cu-tests)
```

- [ ] **Step 4: 下载 stb_image.h**

```bash
curl -o data/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
```

- [ ] **Step 5: 创建 test_patterns.cuh**

生成测试图案的函数声明:

```cpp
#pragma once

#include <cstddef>

struct ImageDimensions {
    size_t width;
    size_t height;
    size_t channels;
};

// 生成棋盘格图案 (alternating black/white squares)
void generateCheckerboard(unsigned char* buffer, size_t width, size_t height, size_t cellSize);

// 生成渐变图案
void generateGradient(unsigned char* buffer, size_t width, size_t height);

// 生成单色图案
void generateSolid(unsigned char* buffer, size_t width, size_t height, unsigned char value);

// 验证两个缓冲区是否相等 (给定容差)
bool compareBuffers(const unsigned char* a, const unsigned char* b, size_t size, float tolerance = 1e-5f);
```

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt data/test_patterns.cuh data/stb_image.h
git commit -m "feat: add project infrastructure for image filters"
```

---

## Task 2: ImageBuffer 基础类和工具函数

**Files:**
- Create: `include/image_utils.h`
- Create: `src/image_utils.cu`

- [ ] **Step 1: 创建 image_utils.h**

```cpp
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <cstddef>
#include <cstdint>

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
    struct Deleter { void operator()(PixelType* p) const { cudaFree(p); } };

public:
    ImageBuffer() : d_data_(nullptr), dims_{0, 0, 0} {}

    ImageBuffer(size_t width, size_t height, size_t channels = 3);

    [[nodiscard]] PixelType* data() const { return d_data_.get(); }
    [[nodiscard]] ImageDimensions dimensions() const { return dims_; }
    [[nodiscard]] size_t size() const { return dims_.width * dims_.height * dims_.channels; }

    void upload(const PixelType* h_data);
    void download(PixelType* h_data) const;
};

template<>
class ImageBuffer<PixelFormat::FLOAT3> {
public:
    using PixelType = float;

private:
    PixelType* d_data_;
    ImageDimensions dims_;
    struct Deleter { void operator()(PixelType* p) const { cudaFree(p); } };

public:
    ImageBuffer() : d_data_(nullptr), dims_{0, 0, 0} {}

    ImageBuffer(size_t width, size_t height, size_t channels = 3);

    [[nodiscard]] PixelType* data() const { return d_data_.get(); }
    [[nodiscard]] ImageDimensions dimensions() const { return dims_; }
    [[nodiscard]] size_t size() const { return dims_.width * dims_.height * dims_.channels; }

    void upload(const PixelType* h_data);
    void download(PixelType* h_data) const;
};

// Helper: Check CUDA error
#define CUDA_CHECK_IMAGE(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
```

- [ ] **Step 2: 创建 image_utils.cu**

实现 ImageBuffer 的构造、upload、download 函数:

```cpp
#include "image_utils.h"

template<PixelFormat Format>
ImageBuffer<Format>::ImageBuffer(size_t width, size_t height, size_t channels)
    : dims_{width, height, channels} {
    CUDA_CHECK_IMAGE(cudaMalloc(&d_data_, size() * sizeof(PixelType)));
}

template<PixelFormat Format>
void ImageBuffer<Format>::upload(const PixelType* h_data) {
    CUDA_CHECK_IMAGE(cudaMemcpy(d_data_.get(), h_data, size() * sizeof(PixelType), cudaMemcpyHostToDevice));
}

template<PixelFormat Format>
void ImageBuffer<Format>::download(PixelType* h_data) const {
    CUDA_CHECK_IMAGE(cudaMemcpy(h_data, d_data_.get(), size() * sizeof(PixelType), cudaMemcpyDeviceToHost));
}

// Explicit instantiations
template class ImageBuffer<PixelFormat::UCHAR3>;
template class ImageBuffer<PixelFormat::FLOAT3>;
```

- [ ] **Step 3: 创建 test_patterns.cuh 实现**

创建 `src/test_patterns.cu`:

```cpp
#include "test_patterns.cuh"
#include <cstring>

void generateCheckerboard(unsigned char* buffer, size_t width, size_t height, size_t cellSize) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            bool cellX = (x / cellSize) % 2 == 0;
            bool cellY = (y / cellSize) % 2 == 0;
            unsigned char value = (cellX == cellY) ? 255 : 0;
            size_t idx = (y * width + x) * 3;
            buffer[idx] = buffer[idx + 1] = buffer[idx + 2] = value;
        }
    }
}

void generateGradient(unsigned char* buffer, size_t width, size_t height) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 3;
            buffer[idx] = static_cast<unsigned char>((x * 255) / width);
            buffer[idx + 1] = static_cast<unsigned char>((y * 255) / height);
            buffer[idx + 2] = static_cast<unsigned char>(((x + y) * 255) / (width + height));
        }
    }
}

void generateSolid(unsigned char* buffer, size_t width, size_t height, unsigned char value) {
    std::memset(buffer, value, width * height * 3);
}

bool compareBuffers(const unsigned char* a, const unsigned char* b, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (diff > tolerance * 255.0f) {
            return false;
        }
    }
    return true;
}
```

- [ ] **Step 4: Commit**

```bash
git add include/image_utils.h src/image_utils.cu src/test_patterns.cu
git commit -m "feat: add ImageBuffer class and test pattern generators"
```

---

## Task 3: Brightness/Contrast Filter

**Files:**
- Create: `include/brightness.h`
- Create: `src/brightness.cu`
- Create: `tests/brightness_test.cu`

- [ ] **Step 1: 创建 brightness.h**

```cpp
#pragma once

#include "image_utils.h"

void adjustBrightnessContrast(const uint8_t* d_input, uint8_t* d_output,
                              size_t width, size_t height,
                              float alpha, float beta);
```

- [ ] **Step 2: 创建 brightness.cu**

```cpp
#include "brightness.h"

__global__ void brightnessContrastKernel(const uint8_t* input, uint8_t* output,
                                         size_t width, size_t height,
                                         float alpha, float beta) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    size_t idx = (y * width + x) * 3;

    for (int c = 0; c < 3; ++c) {
        float value = static_cast<float>(input[idx + c]);
        value = alpha * value + beta;
        value = fminf(255.0f, fmaxf(0.0f, value));
        output[idx + c] = static_cast<uint8_t>(value);
    }
}

void adjustBrightnessContrast(const uint8_t* d_input, uint8_t* d_output,
                              size_t width, size_t height,
                              float alpha, float beta) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    brightnessContrastKernel<<<grid, block>>>(d_input, d_output, width, height, alpha, beta);
    CUDA_CHECK_IMAGE(cudaGetLastError());
    CUDA_CHECK_IMAGE(cudaDeviceSynchronize());
}
```

- [ ] **Step 3: 创建 brightness_test.cu**

```cpp
#include <gtest/gtest.h>
#include "brightness.h"
#include "test_patterns.cuh"

class BrightnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        width_ = 32;
        height_ = 32;
        size_ = width_ * height_ * 3;

        h_input_ = std::make_unique<unsigned char[]>(size_);
        h_output_ = std::make_unique<unsigned char[]>(size_);
        h_expected_ = std::make_unique<unsigned char[]>(size_);

        CUDA_CHECK_IMAGE(cudaMalloc(&d_input_, size_));
        CUDA_CHECK_IMAGE(cudaMalloc(&d_output_, size_));
    }

    void TearDown() override {
        CUDA_CHECK_IMAGE(cudaFree(d_input_));
        CUDA_CHECK_IMAGE(cudaFree(d_output_));
    }

    size_t width_, height_, size_;
    std::unique_ptr<unsigned char[]> h_input_, h_output_, h_expected_;
    uint8_t *d_input_, *d_output_;
};

TEST_F(BrightnessTest, Identity) {
    generateSolid(h_input_.get(), width_, height_, 128);

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));

    adjustBrightnessContrast(d_input_, d_output_, width_, height_, 1.0f, 0.0f);

    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));

    EXPECT_TRUE(compareBuffers(h_input_.get(), h_output_.get(), size_));
}

TEST_F(BrightnessTest, BrightnessIncrease) {
    generateSolid(h_input_.get(), width_, height_, 100);
    std::memset(h_expected_.get(), 150, size_);

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));

    adjustBrightnessContrast(d_input_, d_output_, width_, height_, 1.0f, 50.0f);

    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}

TEST_F(BrightnessTest, ContrastIncrease) {
    generateSolid(h_input_.get(), width_, height_, 128);
    std::memset(h_expected_.get(), 128, size_);

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));

    adjustBrightnessContrast(d_input_, d_output_, width_, height_, 2.0f, 0.0f);

    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));

    EXPECT_TRUE(compareBuffers(h_expected_.get(), h_output_.get(), size_));
}
```

- [ ] **Step 4: 编译测试验证**

```bash
mkdir -p build && cd build && cmake .. && make cu-tests && ./tests/cu-tests
```

- [ ] **Step 5: Commit**

```bash
git add include/brightness.h src/brightness.cu tests/brightness_test.cu
git commit -m "feat: add brightness/contrast filter with tests"
```

---

## Task 4: Gaussian Blur Filter

**Files:**
- Create: `include/gaussian_blur.h`
- Create: `src/gaussian_blur.cu`
- Create: `tests/gaussian_blur_test.cu`

- [ ] **Step 1: 创建 gaussian_blur.h**

```cpp
#pragma once

#include <cstddef>

void gaussianBlur(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  float sigma = 1.0f, int kernel_size = 5);
```

- [ ] **Step 2: 创建 gaussian_blur.cu**

可分离卷积实现:

```cpp
#include "gaussian_blur.h"
#include "cuda_utils.h"

constexpr int MAX_KERNEL_SIZE = 31;

__constant__ float d_kernel_horizontal[MAX_KERNEL_SIZE];
__constant__ float d_kernel_vertical[MAX_KERNEL_SIZE];

__global__ void gaussianBlurHorizontal(const uint8_t* input, float* temp,
                                       size_t width, size_t height,
                                       int kernel_size, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -half; k <= half; ++k) {
        int sx = min(max(static_cast<int>(x) + k, 0), static_cast<int>(width) - 1);
        float weight = d_kernel_horizontal[k + half];
        sum += input[y * width + sx] * weight;
        weight_sum += weight;
    }

    temp[y * width + x] = sum / weight_sum;
}

__global__ void gaussianBlurVertical(const float* temp, uint8_t* output,
                                     size_t width, size_t height,
                                     int kernel_size, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -half; k <= half; ++k) {
        int sy = min(max(static_cast<int>(y) + k, 0), static_cast<int>(height) - 1);
        float weight = d_kernel_vertical[k + half];
        sum += temp[sy * width + x] * weight;
        weight_sum += weight;
    }

    output[y * width + x] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, sum / weight_sum)));
}

void gaussianBlur(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  float sigma, int kernel_size) {
    int half = kernel_size / 2;

    float* h_kernel = new float[kernel_size];
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        int x = i - half;
        h_kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += h_kernel[i];
    }
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] /= sum;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_horizontal, h_kernel, kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_vertical, h_kernel, kernel_size * sizeof(float)));

    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, width * height * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    gaussianBlurHorizontal<<<grid, block>>>(d_input, d_temp, width, height, kernel_size, half);
    CUDA_CHECK(cudaGetLastError());

    gaussianBlurVertical<<<grid, block>>>(d_temp, d_output, width, height, kernel_size, half);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_temp));
    delete[] h_kernel;
}
```

- [ ] **Step 3: 创建 gaussian_blur_test.cu**

测试单色图像模糊后仍为单色,测试边界处理:

```cpp
#include <gtest/gtest.h>
#include "gaussian_blur.h"
#include "test_patterns.cuh"

class GaussianBlurTest : public ::testing::Test {
protected:
    void SetUp() override {
        width_ = 64;
        height_ = 64;
        size_ = width_ * height_ * 3;

        h_input_ = std::make_unique<unsigned char[]>(size_);
        h_output_ = std::make_unique<unsigned char[]>(size_);

        CUDA_CHECK_IMAGE(cudaMalloc(&d_input_, size_));
        CUDA_CHECK_IMAGE(cudaMalloc(&d_output_, size_));
    }

    void TearDown() override {
        CUDA_CHECK_IMAGE(cudaFree(d_input_));
        CUDA_CHECK_IMAGE(cudaFree(d_output_));
    }

    size_t width_, height_, size_;
    std::unique_ptr<unsigned char[]> h_input_, h_output_;
    uint8_t *d_input_, *d_output_;
};

TEST_F(GaussianBlurTest, SolidImage) {
    generateSolid(h_input_.get(), width_, height_, 128);

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));

    gaussianBlur(d_input_, d_output_, width_, height_, 1.0f, 3);

    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 128, 1);
    }
}

TEST_F(GaussianBlurTest, SmallKernel) {
    generateSolid(h_input_.get(), width_, height_, 200);

    CUDA_CHECK_IMAGE(cudaMemcpy(d_input_, h_input_.get(), size_, cudaMemcpyHostToDevice));

    gaussianBlur(d_input_, d_output_, width_, height_, 0.5f, 3);

    CUDA_CHECK_IMAGE(cudaMemcpy(h_output_.get(), d_output_, size_, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output_[i], 200, 2);
    }
}
```

- [ ] **Step 4: 编译测试验证**

- [ ] **Step 5: Commit**

```bash
git add include/gaussian_blur.h src/gaussian_blur.cu tests/gaussian_blur_test.cu
git commit -m "feat: add Gaussian blur filter with separable convolution"
```

---

## Task 5: Sobel Edge Detection

**Files:**
- Create: `include/sobel_edge.h`
- Create: `src/sobel_edge.cu`
- Create: `tests/sobel_edge_test.cu`

- [ ] **Step 1: 创建 sobel_edge.h**

```cpp
#pragma once

#include <cstddef>

void sobelEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                        size_t width, size_t height,
                        float threshold = 30.0f);
```

- [ ] **Step 2: 创建 sobel_edge.cu**

```cpp
#include "sobel_edge.h"
#include "cuda_utils.h"

__global__ void sobelKernel(const uint8_t* input, uint8_t* output,
                            size_t width, size_t height, float threshold) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int gx = -input[(y-1)*width+(x-1)] + input[(y-1)*width+(x+1)]
             -2*input[y*width+(x-1)] + 2*input[y*width+(x+1)]
             -input[(y+1)*width+(x-1)] + input[(y+1)*width+(x+1)];

    int gy = -input[(y-1)*width+(x-1)] - 2*input[(y-1)*width+x] - input[(y-1)*width+(x+1)]
             +input[(y+1)*width+(x-1)] + 2*input[(y+1)*width+x] + input[(y+1)*width+(x+1)];

    int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx*gx + gy*gy)));

    output[y * width + x] = (magnitude > static_cast<int>(threshold)) ? 255 : 0;
}

void sobelEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                        size_t width, size_t height,
                        float threshold) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sobelKernel<<<grid, block>>>(d_input, d_output, width, height, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

- [ ] **Step 3: 创建 sobel_edge_test.cu**

- [ ] **Step 4: 编译测试验证**

- [ ] **Step 5: Commit**

```bash
git add include/sobel_edge.h src/sobel_edge.cu tests/sobel_edge_test.cu
git commit -m "feat: add Sobel edge detection filter"
```

---

## Task 6: 演示程序

**Files:**
- Modify: `src/main.cpp`

- [ ] **Step 1: 更新 main.cpp**

添加演示代码,处理内置图案并输出性能:

```cpp
#include <iostream>
#include <chrono>
#include "brightness.h"
#include "gaussian_blur.h"
#include "sobel_edge.h"
#include "test_patterns.cuh"

void benchmark(const char* name, auto func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << name << ": " << ms << " ms" << std::endl;
}

int main() {
    constexpr size_t WIDTH = 1024;
    constexpr size_t HEIGHT = 1024;
    constexpr size_t SIZE = WIDTH * HEIGHT * 3;

    std::vector<unsigned char> input(SIZE);
    std::vector<unsigned char> output(SIZE);

    generateCheckerboard(input.data(), WIDTH, HEIGHT, 50);

    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, SIZE);
    cudaMalloc(&d_output, SIZE);
    cudaMemcpy(d_input, input.data(), SIZE, cudaMemcpyHostToDevice);

    benchmark("Brightness/Contrast (alpha=1.5, beta=30)", [&]() {
        adjustBrightnessContrast(d_input, d_output, WIDTH, HEIGHT, 1.5f, 30.0f);
    });

    benchmark("Gaussian Blur (sigma=2.0, size=5)", [&]() {
        gaussianBlur(d_input, d_output, WIDTH, HEIGHT, 2.0f, 5);
    });

    benchmark("Sobel Edge Detection", [&]() {
        sobelEdgeDetection(d_input, d_output, WIDTH, HEIGHT, 30.0f);
    });

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Demo complete!" << std::endl;
    return 0;
}
```

- [ ] **Step 2: 编译运行验证**

- [ ] **Step 3: Commit**

```bash
git add src/main.cpp
git commit -m "feat: add image filter demo program"
```

---

## Self-Review Checklist

- [ ] Spec coverage: ImageBuffer ✓, Brightness ✓, Gaussian Blur ✓, Sobel ✓, Tests ✓
- [ ] Placeholder scan: 无 TBD/TODO
- [ ] Type consistency: 所有函数签名一致
- [ ] GoogleTest 测试可运行

---

**Plan complete.** 执行选项:

1. **Subagent-Driven (推荐)** - 逐任务执行,任务间review
2. **Inline Execution** - 本session执行

选哪个?
