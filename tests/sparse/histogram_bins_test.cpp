/**
 * @file histogram_bins_test.cpp
 * @brief Contracts for the histogram bucket count
 *
 * src/parallel/histogram.cu accumulates a 256-entry histogram of uint8_t pixel
 * values: the pixel value IS the bucket index, and HIST_BINS is a compile-time
 * 256 that the kernel's shared array and its `tid < HIST_BINS` guard both use.
 *
 * computeHistogram nevertheless took an `int bins` parameter (defaulting to 256)
 * and honoured it only when clearing memory:
 *
 *   cudaMemset(d_histogram, 0, bins * sizeof(uint32_t));
 *
 * The kernel then wrote every one of the 256 buckets. Measured with bins = 16 and
 * a 256-entry buffer pre-filled with a sentinel:
 *
 *   - one bucket beyond the requested count was written (index 16),
 *   - bucket 100 still held the sentinel, because only 16 of the 256
 *     accumulation threads ran, so pixels with value 100 were never counted.
 *
 * A caller could therefore receive a histogram that was partly cleared, partly
 * accumulated, and silently wrong. The count is now validated: the only shape this
 * kernel can produce is 256 buckets, and asking for another is rejected rather
 * than answered incorrectly.
 */

#include <parallel/histogram.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::parallel::test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

/// Reported histogram size, one bucket per possible uint8_t value.
constexpr size_t kBins = 256;

/// Owns a device buffer.
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) {
        cudaMalloc(&ptr_, bytes);
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const void* host, size_t bytes) {
        cudaMemcpy(ptr_, host, bytes, cudaMemcpyHostToDevice);
    }

    void download(void* host, size_t bytes) const {
        cudaMemcpy(host, ptr_, bytes, cudaMemcpyDeviceToHost);
    }

    template <typename T>
    T* as() { return static_cast<T*>(ptr_); }

private:
    void* ptr_ = nullptr;
};

/// Runs `call` and returns the exception message, or an empty string.
template <typename Callable>
std::string message_from(Callable&& call) {
    try {
        call();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

}  // namespace

// Regression: a bin count other than 256 silently produced a wrong histogram.
TEST(HistogramBinsTest, BinCountOtherThan256IsRejected) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<uint8_t> input(4 * 4 * 3, 100);
    DeviceBuffer device_input(input.size());
    device_input.upload(input.data(), input.size());
    DeviceBuffer histogram(kBins * sizeof(uint32_t));

    const std::string message = message_from([&] {
        computeHistogram(device_input.as<uint8_t>(), histogram.as<uint32_t>(), 4, 4,
                         16);
    });

    EXPECT_FALSE(message.empty())
        << "a 16-bucket request cannot be satisfied by a 256-bucket kernel";
    EXPECT_NE(message.find("256"), std::string::npos)
        << "the message should say what the kernel actually produces: " << message;
    EXPECT_NE(message.find("16"), std::string::npos)
        << "the message should echo what was requested: " << message;
}

TEST(HistogramBinsTest, Explicit256IsAccepted) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<uint8_t> input(2 * 2 * 3, 7);
    DeviceBuffer device_input(input.size());
    device_input.upload(input.data(), input.size());
    DeviceBuffer histogram(kBins * sizeof(uint32_t));

    EXPECT_NO_THROW(computeHistogram(device_input.as<uint8_t>(),
                                     histogram.as<uint32_t>(), 2, 2, 256));
}

TEST(HistogramBinsTest, CountsEveryPixelOfAUniformImage) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A uniform grey image: every one of the width*height*3 channel samples has
    // the same value, so one bucket holds all of them and the rest stay zero.
    constexpr size_t width = 4, height = 4;
    const uint8_t value = 100;
    const std::vector<uint8_t> input(width * height * 3, value);

    DeviceBuffer device_input(input.size());
    device_input.upload(input.data(), input.size());
    DeviceBuffer histogram(kBins * sizeof(uint32_t));

    computeHistogram(device_input.as<uint8_t>(), histogram.as<uint32_t>(), width,
                     height);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> host(kBins, 0);
    histogram.download(host.data(), kBins * sizeof(uint32_t));

    // Three interleaved channels, so every sample lands in the same bucket.
    EXPECT_EQ(host[value], width * height * 3)
        << "the bucket for value " << static_cast<int>(value) << " is wrong";
    EXPECT_EQ(host[0], 0u) << "an unrelated bucket must be cleared and stay empty";
    EXPECT_EQ(host[255], 0u);
}

TEST(HistogramBinsTest, ClearsTheWholeBuffer) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Pre-fill with a sentinel: any bucket the kernel neither clears nor
    // increments keeps the sentinel, which is how the old bug was visible.
    const std::vector<uint32_t> sentinel(kBins, 0xAAAAAAAAu);
    const std::vector<uint8_t> input(2 * 2 * 3, 200);

    DeviceBuffer histogram(kBins * sizeof(uint32_t));
    histogram.upload(sentinel.data(), kBins * sizeof(uint32_t));
    DeviceBuffer device_input(input.size());
    device_input.upload(input.data(), input.size());

    computeHistogram(device_input.as<uint8_t>(), histogram.as<uint32_t>(), 2, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> host(kBins, 0);
    histogram.download(host.data(), kBins * sizeof(uint32_t));

    for (size_t i = 0; i < kBins; ++i) {
        EXPECT_NE(host[i], 0xAAAAAAAAu)
            << "bucket " << i << " was never touched, so the buffer was not fully "
                                 "cleared and not fully accumulated";
    }
    EXPECT_EQ(host[200], 2u * 2 * 3) << "the populated bucket has the wrong count";
}

TEST(HistogramBinsTest, PerChannelUsesTheSameBucketCount) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A pure red image: the red channel accumulates everything into bucket 255,
    // green and blue into bucket 0. This also pins that the per-channel entry
    // point does not take a bin count at all, so they cannot disagree.
    constexpr size_t width = 2, height = 2;
    std::vector<uint8_t> input(width * height * 3, 0);
    for (size_t i = 0; i < width * height; ++i) {
        input[i * 3 + 0] = 255;  // red
        input[i * 3 + 1] = 0;    // green
        input[i * 3 + 2] = 0;    // blue
    }

    DeviceBuffer device_input(input.size());
    device_input.upload(input.data(), input.size());
    DeviceBuffer red(kBins * sizeof(uint32_t));
    DeviceBuffer green(kBins * sizeof(uint32_t));
    DeviceBuffer blue(kBins * sizeof(uint32_t));

    computeHistogramPerChannel(device_input.as<uint8_t>(), red.as<uint32_t>(),
                               green.as<uint32_t>(), blue.as<uint32_t>(), width,
                               height);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> host_red(kBins, 0), host_green(kBins, 0),
        host_blue(kBins, 0);
    red.download(host_red.data(), kBins * sizeof(uint32_t));
    green.download(host_green.data(), kBins * sizeof(uint32_t));
    blue.download(host_blue.data(), kBins * sizeof(uint32_t));

    EXPECT_EQ(host_red[255], width * height);
    EXPECT_EQ(host_green[0], width * height);
    EXPECT_EQ(host_blue[0], width * height);
}

}  // namespace cuda::parallel::test
