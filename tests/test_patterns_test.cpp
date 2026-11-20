#include <gtest/gtest.h>
#include "test_patterns.cuh"
#include <vector>
#include <cmath>

class TestPatternsTest : public ::testing::Test {
protected:
    static constexpr size_t SMALL = 8;
    static constexpr size_t MEDIUM = 64;
};

TEST_F(TestPatternsTest, GenerateSolidBlack) {
    std::vector<unsigned char> buffer(SMALL * SMALL * 3, 0xFF);
    generateSolid(buffer.data(), SMALL, SMALL, 0);

    for (unsigned char val : buffer) {
        EXPECT_EQ(val, 0);
    }
}

TEST_F(TestPatternsTest, GenerateSolidWhite) {
    std::vector<unsigned char> buffer(SMALL * SMALL * 3, 0);
    generateSolid(buffer.data(), SMALL, SMALL, 255);

    for (unsigned char val : buffer) {
        EXPECT_EQ(val, 255);
    }
}

TEST_F(TestPatternsTest, GenerateSolidGray) {
    constexpr unsigned char GRAY = 128;
    std::vector<unsigned char> buffer(SMALL * SMALL * 3, 0);
    generateSolid(buffer.data(), SMALL, SMALL, GRAY);

    for (unsigned char val : buffer) {
        EXPECT_EQ(val, GRAY);
    }
}

TEST_F(TestPatternsTest, CheckerboardAlternating) {
    constexpr size_t cellSize = 2;
    std::vector<unsigned char> buffer(SMALL * SMALL * 3, 0);
    generateCheckerboard(buffer.data(), SMALL, SMALL, cellSize);

    // (0,0): cellX=0, cellY=0, same -> 255
    EXPECT_EQ(buffer[0], 255);
    // (1,0): cellX=0, cellY=0, same -> 255
    EXPECT_EQ(buffer[3], 255);
    // (2,0): cellX=1, cellY=0, diff -> 0
    EXPECT_EQ(buffer[6], 0);
}

TEST_F(TestPatternsTest, CheckerboardCellSizeOne) {
    constexpr size_t cellSize = 1;
    std::vector<unsigned char> buffer(4 * 4 * 3, 0);
    generateCheckerboard(buffer.data(), 4, 4, cellSize);

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            bool expectedWhite = (x % 2) == (y % 2);
            unsigned char expected = expectedWhite ? 255 : 0;
            size_t idx = (y * 4 + x) * 3;
            EXPECT_EQ(buffer[idx], expected) << "Failed at (" << x << "," << y << ")";
        }
    }
}

TEST_F(TestPatternsTest, CheckerboardLargeCells) {
    constexpr size_t cellSize = 10;
    std::vector<unsigned char> buffer(20 * 20 * 3, 0);
    generateCheckerboard(buffer.data(), 20, 20, cellSize);

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            size_t idx = (y * 20 + x) * 3;
            EXPECT_EQ(buffer[idx], 255) << "Failed at first cell";
        }
    }
}

TEST_F(TestPatternsTest, GradientMonotonic) {
    std::vector<unsigned char> buffer(MEDIUM * MEDIUM * 3, 0);
    generateGradient(buffer.data(), MEDIUM, MEDIUM);

    for (size_t x = 1; x + 1 < static_cast<size_t>(MEDIUM); ++x) {
        size_t idx0 = x * 3;
        size_t idx1 = (x + 1) * 3;
        EXPECT_GE(buffer[idx1], buffer[idx0]) << "R not monotonic at x=" << x;
    }

    for (size_t y = 1; y + 1 < static_cast<size_t>(MEDIUM); ++y) {
        size_t idx0 = (y - 1) * MEDIUM * 3;
        size_t idx1 = y * MEDIUM * 3 + 1;
        EXPECT_GE(buffer[idx1], buffer[idx0]) << "G not monotonic at y=" << y;
    }
}

TEST_F(TestPatternsTest, GradientValueRange) {
    std::vector<unsigned char> buffer(MEDIUM * MEDIUM * 3, 0);
    generateGradient(buffer.data(), MEDIUM, MEDIUM);

    for (unsigned char val : buffer) {
        EXPECT_GE(val, 0);
        EXPECT_LE(val, 255);
    }
}

TEST_F(TestPatternsTest, CompareBuffersIdentical) {
    std::vector<unsigned char> a(MEDIUM * MEDIUM * 3, 128);
    std::vector<unsigned char> b(MEDIUM * MEDIUM * 3, 128);

    EXPECT_TRUE(compareBuffers(a.data(), b.data(), a.size()));
}

TEST_F(TestPatternsTest, CompareBuffersDifferent) {
    std::vector<unsigned char> a(SMALL * SMALL * 3, 0);
    std::vector<unsigned char> b(SMALL * SMALL * 3, 255);

    EXPECT_FALSE(compareBuffers(a.data(), b.data(), a.size()));
}

TEST_F(TestPatternsTest, CompareBuffersWithTolerance) {
    std::vector<unsigned char> a = {100};
    std::vector<unsigned char> b = {101};

    EXPECT_TRUE(compareBuffers(a.data(), b.data(), 1, 0.01f));
}

TEST_F(TestPatternsTest, CompareBuffersOutsideTolerance) {
    std::vector<unsigned char> a = {100};
    std::vector<unsigned char> b = {110};

    EXPECT_FALSE(compareBuffers(a.data(), b.data(), 1, 0.01f));
}

TEST_F(TestPatternsTest, GenerateCheckerboardOddDimensions) {
    std::vector<unsigned char> buffer(7 * 7 * 3, 0);
    generateCheckerboard(buffer.data(), 7, 7, 2);

    for (size_t i = 0; i < 7 * 7 * 3; ++i) {
        EXPECT_TRUE(buffer[i] == 0 || buffer[i] == 255);
    }
}

TEST_F(TestPatternsTest, GenerateGradientSinglePixel) {
    std::vector<unsigned char> buffer(1 * 1 * 3, 0);
    generateGradient(buffer.data(), 1, 1);

    EXPECT_GE(buffer[0], 0);
    EXPECT_LE(buffer[0], 255);
}
