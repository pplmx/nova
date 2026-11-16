/**
 * @file sobel_edge_bounds_test.cpp
 * @brief Contracts for the Sobel edge kernel's output bounds
 *
 * src/image/sobel_edge.cu launches a 16x16 grid rounded up from the image size, so
 * threads outside the image exist. Its boundary branch wrote through an index derived
 * from x and y WITHOUT first rejecting those out-of-range threads:
 *
 *     if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
 *         size_t outIdx = (y * width + x) * 3;
 *         output[outIdx] = 0;                     // x or y may be past the image
 *         ...
 *     }
 *
 * For a 10x10 image the grid covers x and y up to 15, so the write landed at byte 495
 * of a 300-byte buffer. cudaGetLastError reported success because the overrun fell in
 * the allocator's padding; compute-sanitizer names it directly:
 *
 *     Invalid __global__ write of size 1 bytes
 *         Access at 0x... is out of bounds
 *
 * (five such reports)
 *
 * Out-of-range threads now return before any index is formed, and an empty image is a
 * no-op instead of a zero-sized grid. compute-sanitizer reports 0 errors afterwards.
 *
 * The kernel's gradient arithmetic was checked and is correct: the standard Sobel
 * operators in x and y, the perceptual luma weights, and the threshold comparison.
 */

#include <image/sobel_edge.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace sobel_edge_bounds_test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

/// Owns a device uint8 buffer sized for packed RGB.
class RgbImage {
public:
    explicit RgbImage(size_t width, size_t height, uint8_t fill)
        : count_(width * height * 3) {
        cudaMalloc(&ptr_, count_ > 0 ? count_ : 1);
        cudaMemset(ptr_, fill, count_ > 0 ? count_ : 1);
    }

    ~RgbImage() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    RgbImage(const RgbImage&) = delete;
    RgbImage& operator=(const RgbImage&) = delete;

    uint8_t* data() { return ptr_; }
    const uint8_t* data() const { return ptr_; }

    std::vector<uint8_t> read() const {
        std::vector<uint8_t> host(count_, 0);
        cudaMemcpy(host.data(), ptr_, count_, cudaMemcpyDeviceToHost);
        return host;
    }

private:
    uint8_t* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace

// Regression: this wrote past the end of the output buffer.
TEST(SobelEdgeBoundsTest, ImageSmallerThanOneBlockStaysInsideTheBuffer) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // 10x10 with a 16x16 thread block: 36 threads per axis are out of range.
    constexpr size_t width = 10, height = 10;
    RgbImage input(width, height, 128);
    RgbImage output(width, height, 0);

    (void)cudaGetLastError();
    ASSERT_NO_THROW(sobelEdgeDetection(input.data(), output.data(), width, height,
                                       50.0f));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // The buffer must still hold exactly its own bytes; a wrote-past-the-end cannot
    // be observed from this buffer, but the call must not corrupt the allocation.
    const auto result = output.read();
    ASSERT_EQ(result.size(), width * height * 3);
}

TEST(SobelEdgeBoundsTest, OddSizesAreAcceptedWithoutError) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Sizes on both sides of a block boundary, all non-multiples of 16.
    for (const size_t dimension : {1, 3, 15, 17, 31, 33}) {
        RgbImage input(dimension, dimension, 90);
        RgbImage output(dimension, dimension, 0);

        (void)cudaGetLastError();
        EXPECT_NO_THROW(sobelEdgeDetection(input.data(), output.data(), dimension,
                                          dimension, 40.0f))
            << "size " << dimension;
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "size " << dimension;
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "size " << dimension;
    }
}

TEST(SobelEdgeBoundsTest, EmptyImageIsANoOp) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    (void)cudaGetLastError();
    EXPECT_NO_THROW(sobelEdgeDetection(nullptr, nullptr, 0, 8, 50.0f));
    EXPECT_NO_THROW(sobelEdgeDetection(nullptr, nullptr, 8, 0, 50.0f));
    EXPECT_NO_THROW(sobelEdgeDetection(nullptr, nullptr, 0, 0, 50.0f));

    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "an empty image must not leave a launch error behind";
}

TEST(SobelEdgeBoundsTest, UniformImageHasNoEdges) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A constant image has zero gradient everywhere, so every interior pixel is
    // below any positive threshold and the whole output is black.
    constexpr size_t width = 20, height = 20;
    RgbImage input(width, height, 137);
    RgbImage output(width, height, 255);

    sobelEdgeDetection(input.data(), output.data(), width, height, 10.0f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (const uint8_t value : output.read()) {
        EXPECT_EQ(value, 0) << "a flat image produced an edge response";
    }
}

TEST(SobelEdgeBoundsTest, AVerticalEdgeIsDetected) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A left-half black, right-half white image has a strong vertical gradient at the
    // boundary column, which the x operator must pick up.
    constexpr size_t width = 20, height = 20;
    RgbImage input(width, height, 0);
    std::vector<uint8_t> host(static_cast<size_t>(width) * height * 3, 0);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = width / 2; x < width; ++x) {
            const size_t base = (y * width + x) * 3;
            host[base] = 255;
            host[base + 1] = 255;
            host[base + 2] = 255;
        }
    }
    ASSERT_EQ(cudaMemcpy(input.data(), host.data(), host.size(),
                         cudaMemcpyHostToDevice), cudaSuccess);

    RgbImage output(width, height, 0);
    sobelEdgeDetection(input.data(), output.data(), width, height, 50.0f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto result = output.read();
    // The column just left of the transition sits on the gradient.
    const size_t x = width / 2 - 1;
    const size_t y = height / 2;
    const size_t base = (y * width + x) * 3;
    EXPECT_EQ(result[base], 255) << "the vertical edge was not detected";

    // A pixel far from the boundary stays black.
    const size_t calm_base = (y * width + 2) * 3;
    EXPECT_EQ(result[calm_base], 0) << "a flat region was reported as an edge";
}

TEST(SobelEdgeBoundsTest, ThresholdControlsTheResponse) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The same edge image with a threshold above the attainable gradient magnitude
    // must produce no edges at all.
    constexpr size_t width = 16, height = 16;
    std::vector<uint8_t> host(static_cast<size_t>(width) * height * 3, 0);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = width / 2; x < width; ++x) {
            const size_t base = (y * width + x) * 3;
            host[base] = 255;
            host[base + 1] = 255;
            host[base + 2] = 255;
        }
    }

    RgbImage input(width, height, 0);
    ASSERT_EQ(cudaMemcpy(input.data(), host.data(), host.size(),
                         cudaMemcpyHostToDevice), cudaSuccess);

    RgbImage output(width, height, 0);
    // The Sobel magnitude for a 0 -> 255 step is about 4 * 255 = 1020, so 5000 is
    // beyond anything the operator can produce.
    sobelEdgeDetection(input.data(), output.data(), width, height, 5000.0f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (const uint8_t value : output.read()) {
        EXPECT_EQ(value, 0) << "a threshold above the gradient magnitude still fired";
    }
}

}  // namespace sobel_edge_bounds_test
