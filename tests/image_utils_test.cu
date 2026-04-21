#include <gtest/gtest.h>
#include "image_utils.h"
#include "test_patterns.cuh"
#include <vector>

class ImageBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        width_ = 32;
        height_ = 32;
        size_ = width_ * height_ * 3;
    }

    size_t width_;
    size_t height_;
    size_t size_;
};

TEST_F(ImageBufferTest, DefaultConstruction) {
    ImageBuffer<PixelFormat::UCHAR3> buffer;
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.dimensions().width, 0);
    EXPECT_EQ(buffer.dimensions().height, 0);
    EXPECT_FALSE(static_cast<bool>(buffer));
}

TEST_F(ImageBufferTest, ConstructionWithSize) {
    ImageBuffer<PixelFormat::UCHAR3> buffer(width_, height_);
    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.dimensions().width, width_);
    EXPECT_EQ(buffer.dimensions().height, height_);
    EXPECT_EQ(buffer.dimensions().channels, 3);
    EXPECT_TRUE(static_cast<bool>(buffer));
}

TEST_F(ImageBufferTest, UploadDownload) {
    ImageBuffer<PixelFormat::UCHAR3> buffer(width_, height_);

    std::vector<uint8_t> h_input(size_);
    generateSolid(h_input.data(), width_, height_, 128);

    std::vector<uint8_t> h_output(size_, 0);

    buffer.upload(h_input.data());
    buffer.download(h_output.data());

    EXPECT_TRUE(compareBuffers(h_input.data(), h_output.data(), size_));
}

TEST_F(ImageBufferTest, FloatVersion) {
    ImageBuffer<PixelFormat::FLOAT3> buffer(width_, height_);

    std::vector<float> h_input(size_, 0.5f);
    std::vector<float> h_output(size_, 0.0f);

    buffer.upload(h_input.data());
    buffer.download(h_output.data());

    for (size_t i = 0; i < size_; ++i) {
        EXPECT_NEAR(h_output[i], 0.5f, 1e-5f);
    }
}

TEST_F(ImageBufferTest, ZeroSizeBuffer) {
    ImageBuffer<PixelFormat::UCHAR3> buffer(0, 0);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.data(), nullptr);
}
