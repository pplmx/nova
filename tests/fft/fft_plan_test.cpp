#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "cuda/fft/fft.h"
#include "cuda/stream/stream.h"

class FFTPlanTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(FFTPlanTest, ConstructionWithSize) {
    cuda::fft::FFTPlan plan(256);
    EXPECT_TRUE(static_cast<bool>(plan));
    EXPECT_EQ(plan.size(), 256);
}

TEST_F(FFTPlanTest, ConstructionWithDifferentSizes) {
    cuda::fft::FFTPlan plan1(512);
    cuda::fft::FFTPlan plan2(1024);
    EXPECT_TRUE(static_cast<bool>(plan1));
    EXPECT_TRUE(static_cast<bool>(plan2));
    EXPECT_EQ(plan1.size(), 512);
    EXPECT_EQ(plan2.size(), 1024);
}

TEST_F(FFTPlanTest, ConstructionWithDirection) {
    cuda::fft::FFTPlan forward_plan(256, cuda::fft::Direction::Forward);
    cuda::fft::FFTPlan inverse_plan(256, cuda::fft::Direction::Inverse);
    EXPECT_TRUE(static_cast<bool>(forward_plan));
    EXPECT_TRUE(static_cast<bool>(inverse_plan));
}

TEST_F(FFTPlanTest, Construction2D) {
    cuda::fft::FFTPlan plan(64, 64);
    EXPECT_TRUE(static_cast<bool>(plan));
    EXPECT_EQ(plan.size(), 64 * 64);
}

TEST_F(FFTPlanTest, Construction3D) {
    cuda::fft::FFTPlan plan(8, 8, 8);
    EXPECT_TRUE(static_cast<bool>(plan));
    EXPECT_EQ(plan.size(), 8 * 8 * 8);
}

TEST_F(FFTPlanTest, MoveConstructor) {
    cuda::fft::FFTPlan plan1(256);
    auto handle1 = plan1.handle();
    cuda::fft::FFTPlan plan2(std::move(plan1));
    EXPECT_TRUE(static_cast<bool>(plan2));
    EXPECT_EQ(plan2.handle(), handle1);
}

TEST_F(FFTPlanTest, MoveAssignment) {
    cuda::fft::FFTPlan plan1(256);
    cuda::fft::FFTPlan plan2(512);
    auto handle1 = plan1.handle();
    plan2 = std::move(plan1);
    EXPECT_TRUE(static_cast<bool>(plan2));
    EXPECT_EQ(plan2.handle(), handle1);
}

TEST_F(FFTPlanTest, HandleAccess) {
    cuda::fft::FFTPlan plan(128);
    EXPECT_NE(plan.handle(), 0);
}

TEST_F(FFTPlanTest, MultiplePlansCanExist) {
    std::vector<cuda::fft::FFTPlan> plans;
    for (size_t i = 64; i <= 512; i *= 2) {
        plans.emplace_back(i);
    }
    for (const auto& plan : plans) {
        EXPECT_TRUE(static_cast<bool>(plan));
    }
}
