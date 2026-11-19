/**
 * @file image_morphology_edge_test.cpp
 * @brief Contracts for the image filters' extents and channel layout
 *
 * src/image/morphology.cu provides sharpen, threshold, erode and dilate. Two things
 * were wrong.
 *
 * - Every launcher derived its grid from width and height without checking them, so a
 *   zero extent made one grid dimension zero:
 *
 *       sharpenImage(width=0)          -> CUDA error: invalid configuration argument
 *       sharpenImage(height=0)         -> the same
 *       applyThreshold(width=0)        -> the same
 *       erodeImage(width=0)            -> the same
 *
 *   A zero extent is an ordinary request; all four now return without launching.
 *
 * - The kernels index packed RGB — (y * width + x) * 3 — but the header said only
 *   "width, height" and never stated the element count. Measured: a buffer allocated
 *   as width * height bytes was read up to byte 192 of 64, with cudaGetLastError
 *   reporting success because the overrun fell inside the allocator's padding. A
 *   silent over-read is worse than a crash, so the header now documents the packing.
 */

#include <image/brightness.h>
#include <image/morphology.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace image_morphology_edge_test {

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

    // Regression: all four threw "invalid configuration argument".
    TEST(ImageMorphologyEdgeTest, ZeroExtentsAreNoOps) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        (void)cudaGetLastError();

        EXPECT_NO_THROW(sharpenImage(nullptr, nullptr, 0, 8, 1.0f));
        EXPECT_NO_THROW(sharpenImage(nullptr, nullptr, 8, 0, 1.0f));
        EXPECT_NO_THROW(applyThreshold(nullptr, nullptr, 0, 8, 128));
        EXPECT_NO_THROW(erodeImage(nullptr, nullptr, 0, 8, 3));
        EXPECT_NO_THROW(dilateImage(nullptr, nullptr, 0, 8, 3));
        EXPECT_NO_THROW(erodeImage(nullptr, nullptr, 8, 0, 3));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "an empty image must not leave a launch error behind";
    }

    TEST(ImageMorphologyEdgeTest, BuffersArePackedRgb) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The documented element count is width * height * 3. A uniform image survives
        // sharpen, threshold, erode and dilate unchanged, so a wrongly-sized buffer is
        // the only thing that can go wrong here — and the earlier code would have read
        // past it without saying so.
        constexpr size_t width = 8, height = 8;
        const uint8_t fill = 100;

        RgbImage input(width, height, fill);
        RgbImage output(width, height, 0);

        sharpenImage(input.data(), output.data(), width, height, 1.0f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "no over-read may have happened";

        const auto sharpened = output.read();
        ASSERT_EQ(sharpened.size(), width * height * 3);

        // sharpenKernel only writes interior pixels (it skips x or y < 1 and the last
        // row and column), so the border keeps whatever the output buffer started with.
        // What must hold is that an interior pixel of a uniform image is unchanged: its
        // Laplacian is zero. My first version of this case asserted the whole buffer and
        // failed on the border, which is documented kernel behaviour rather than a defect.
        for (size_t y = 1; y + 1 < height; ++y) {
            for (size_t x = 1; x + 1 < width; ++x) {
                const size_t base = (y * width + x) * 3;
                for (size_t c = 0; c < 3; ++c) {
                    EXPECT_EQ(sharpened[base + c], fill) << "interior pixel (" << x << "," << y << ") channel " << c;
                }
            }
        }
    }

    // Regression: the border of a sharpened image used to hold uninitialised memory.
    TEST(ImageMorphologyEdgeTest, SharpenedBorderHoldsTheOriginalPixels) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The stencil cannot be evaluated on the first or last row and column, so those
        // pixels are left as they were. Before the output was seeded from the input they
        // held whatever the allocation contained: compute-sanitizer --tool initcheck
        // reported 66 uninitialized accesses when a 64x48 result was copied back to the
        // host. The existing SharpenOutputInRange case could not see this because
        // EXPECT_GE on a uint8_t and EXPECT_LE against 255 are both always true.
        constexpr size_t width = 16, height = 12;
        RgbImage input(width, height, 0);

        // A gradient so every pixel differs from its neighbours.
        std::vector<uint8_t> host(static_cast<size_t>(width) * height * 3, 0);
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = static_cast<uint8_t>((i * 7) % 200 + 20);
        }
        ASSERT_EQ(cudaMemcpy(input.data(), host.data(), host.size(), cudaMemcpyHostToDevice), cudaSuccess);

        RgbImage output(width, height, 0xFF);  // poison, so an untouched byte shows up
        sharpenImage(input.data(), output.data(), width, height, 1.0f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), host.size());

        const auto index_of = [width](size_t x, size_t y) { return (y * width + x) * 3; };

        for (size_t x = 0; x < width; ++x) {
            for (const size_t y : {size_t{0}, height - 1}) {
                const size_t base = index_of(x, y);
                for (size_t c = 0; c < 3; ++c) {
                    EXPECT_EQ(result[base + c], host[base + c]) << "border pixel (" << x << "," << y << ") channel " << c;
                }
            }
        }
        for (size_t y = 0; y < height; ++y) {
            for (const size_t x : {size_t{0}, width - 1}) {
                const size_t base = index_of(x, y);
                for (size_t c = 0; c < 3; ++c) {
                    EXPECT_EQ(result[base + c], host[base + c]) << "border pixel (" << x << "," << y << ") channel " << c;
                }
            }
        }
    }

    TEST(ImageMorphologyEdgeTest, SharpeningAUniformImageIsTheIdentity) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The Laplacian is zero on a flat region, so the whole image must come back
        // unchanged at any strength.
        constexpr size_t width = 12, height = 12;
        const uint8_t fill = 60;

        RgbImage input(width, height, fill);
        RgbImage output(width, height, 0xFF);

        sharpenImage(input.data(), output.data(), width, height, 2.5f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        for (const uint8_t value : output.read()) {
            EXPECT_EQ(value, fill);
        }
    }

    TEST(ImageMorphologyEdgeTest, OpeningAndClosingAreComposites) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Both are defined in terms of the two primitives: opening is erode-then-dilate,
        // closing is dilate-then-erode. Checking the composite against the primitives
        // applied by hand is the only way to tell an implemented composition from a
        // plausible-looking one.
        constexpr size_t width = 10, height = 10;
        const uint8_t fill = 120;

        RgbImage input(width, height, fill);
        RgbImage staged(width, height, 0);
        RgbImage expected(width, height, 0);
        RgbImage actual(width, height, 0);

        // Reference: apply the two steps explicitly.
        erodeImage(input.data(), staged.data(), width, height, 3);
        dilateImage(staged.data(), expected.data(), width, height, 3);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        openingImage(input.data(), actual.data(), width, height, 3);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        EXPECT_EQ(actual.read(), expected.read()) << "openingImage must be erode followed by dilate";

        // And closing the other way round.
        dilateImage(input.data(), staged.data(), width, height, 3);
        erodeImage(staged.data(), expected.data(), width, height, 3);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        closingImage(input.data(), actual.data(), width, height, 3);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        EXPECT_EQ(actual.read(), expected.read()) << "closingImage must be dilate followed by erode";
    }

    TEST(ImageMorphologyEdgeTest, OpeningAndClosingHandleEmptyExtents) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // These two call cudaMalloc for an intermediate buffer, so an empty extent used to
        // reach cudaMalloc(0) before reaching the guarded primitives.
        (void)cudaGetLastError();

        EXPECT_NO_THROW(openingImage(nullptr, nullptr, 0, 8, 3));
        EXPECT_NO_THROW(openingImage(nullptr, nullptr, 8, 0, 3));
        EXPECT_NO_THROW(closingImage(nullptr, nullptr, 0, 0, 3));
        EXPECT_NO_THROW(closingImage(nullptr, nullptr, 4, 0, 3));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

    // A case asserting that repeated composites do not leak was written and then removed:
    // it compared cudaMemGetInfo before and after, and after injecting a deliberate leak
    // into openingImage it STILL PASSED. The CUDA allocator keeps freed blocks in its pool,
    // so the free figure does not move for a small per-call leak — the same reason a leak
    // check of this shape failed in an earlier round on a different module. Measuring this
    // properly needs a tool that tracks allocations, not a device-wide snapshot, and that
    // measurement belongs in a dedicated leak-check suite rather than here.

    // Regression: this was the one function in src/image/ without an empty-extent guard.
    TEST(ImageMorphologyEdgeTest, BrightnessAcceptsEmptyExtents) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A zero extent produced a zero-sized grid dimension, which CUDA rejects with a
        // generic "invalid configuration argument" — measured for both width == 0 and
        // height == 0. Every other image entry point already returned early.
        (void)cudaGetLastError();

        EXPECT_NO_THROW(adjustBrightnessContrast(nullptr, nullptr, 0, 8, 1.0f, 0.0f));
        EXPECT_NO_THROW(adjustBrightnessContrast(nullptr, nullptr, 8, 0, 1.0f, 0.0f));
        EXPECT_NO_THROW(adjustBrightnessContrast(nullptr, nullptr, 0, 0, 1.0f, 0.0f));

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "an empty image must not leave a launch error behind";
    }

    TEST(ImageMorphologyEdgeTest, BrightnessAppliesContrastThenOffset) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // out = clamp(contrast * in + offset). A 2x2 image (packed RGB) so the channel
        // layout is exercised alongside the arithmetic.
        constexpr size_t width = 2, height = 2;
        const std::vector<uint8_t> input = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        ASSERT_EQ(cudaMalloc(&d_input, input.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, input.size()), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice), cudaSuccess);

        adjustBrightnessContrast(d_input, d_output, width, height, 2.0f, 10.0f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<uint8_t> result(input.size(), 0);
        ASSERT_EQ(cudaMemcpy(result.data(), d_output, input.size(), cudaMemcpyDeviceToHost), cudaSuccess);

        for (size_t i = 0; i < input.size(); ++i) {
            const int expected = std::min(255, std::max(0, int(input[i]) * 2 + 10));
            EXPECT_EQ(result[i], expected) << "byte " << i << " held " << int(input[i]);
        }

        cudaFree(d_input);
        cudaFree(d_output);
    }

    TEST(ImageMorphologyEdgeTest, BrightnessClampsInsteadOfWrapping) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 200 * 3 + 100 overflows a byte if the kernel casts without clamping; the result
        // must saturate at 255 rather than wrap around.
        constexpr size_t width = 1, height = 1;
        const std::vector<uint8_t> input = {200, 200, 200};

        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        ASSERT_EQ(cudaMalloc(&d_input, 3), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, 3), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, input.data(), 3, cudaMemcpyHostToDevice), cudaSuccess);

        adjustBrightnessContrast(d_input, d_output, width, height, 3.0f, 100.0f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<uint8_t> result(3, 0);
        ASSERT_EQ(cudaMemcpy(result.data(), d_output, 3, cudaMemcpyDeviceToHost), cudaSuccess);
        for (const uint8_t value : result) {
            EXPECT_EQ(value, 255) << "an overflowing result must saturate";
        }

        cudaFree(d_input);
        cudaFree(d_output);
    }

    TEST(ImageMorphologyEdgeTest, ThresholdSplitsOnTheGivenValue) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Two bytes below the threshold and two at it: the split must be consistent
        // across the whole packed buffer.
        constexpr size_t width = 2, height = 1;
        const std::vector<uint8_t> host = {0, 50, 200, 255, 100, 99};

        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;
        ASSERT_EQ(cudaMalloc(&d_input, host.size()), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, host.size()), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, host.data(), host.size(), cudaMemcpyHostToDevice), cudaSuccess);

        applyThreshold(d_input, d_output, width, height, 100);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<uint8_t> result(host.size(), 0);
        ASSERT_EQ(cudaMemcpy(result.data(), d_output, host.size(), cudaMemcpyDeviceToHost), cudaSuccess);

        // The kernel compares with strict greater-than, so a byte exactly at the
        // threshold becomes 0. That is a legitimate choice for a binarisation threshold;
        // my first version assumed >= and failed on the two bytes equal to 100.
        for (size_t i = 0; i < host.size(); ++i) {
            const uint8_t expected = host[i] > 100 ? 255 : 0;
            EXPECT_EQ(result[i], expected) << "byte " << i << " held " << int(host[i]);
        }

        cudaFree(d_input);
        cudaFree(d_output);
    }

    TEST(ImageMorphologyEdgeTest, ErodeAndDilateLeaveAUniformImageUnchanged) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Every neighbourhood of a uniform image is the same value, so min and max are
        // both that value.
        constexpr size_t width = 6, height = 6;
        const uint8_t fill = 77;

        RgbImage input(width, height, fill);
        RgbImage eroded(width, height, 0);
        RgbImage dilated(width, height, 0);

        erodeImage(input.data(), eroded.data(), width, height, 3);
        dilateImage(input.data(), dilated.data(), width, height, 3);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        for (const uint8_t value : eroded.read()) {
            EXPECT_EQ(value, fill);
        }
        for (const uint8_t value : dilated.read()) {
            EXPECT_EQ(value, fill);
        }
    }

    TEST(ImageMorphologyEdgeTest, SinglePixelImageIsAccepted) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The smallest non-empty image: the boundary clamping in the neighbourhood loops
        // must handle it.
        RgbImage input(1, 1, 42);
        RgbImage output(1, 1, 0);

        EXPECT_NO_THROW(erodeImage(input.data(), output.data(), 1, 1, 3));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto result = output.read();
        ASSERT_EQ(result.size(), 3u);
        for (const uint8_t value : result) {
            EXPECT_EQ(value, 42) << "a 1x1 image is its own neighbourhood";
        }
    }

}  // namespace image_morphology_edge_test
