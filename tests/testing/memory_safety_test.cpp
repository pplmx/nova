#include <gtest/gtest.h>

#include "cuda/testing/memory_safety.h"

namespace cuda::testing::test {

class MemorySafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        MemorySafetyValidator::instance().reset();
        MemorySafetyValidator::instance().enable();
    }
};

TEST_F(MemorySafetyTest, ValidateNullptrReturnsFalse) {
    auto result = MemorySafetyValidator::instance().validate_allocation(nullptr, 100);
    EXPECT_FALSE(result);
    EXPECT_EQ(MemorySafetyValidator::instance().error_count(), 1);
}

TEST_F(MemorySafetyTest, ValidateValidAllocation) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, 1024);

    auto result = MemorySafetyValidator::instance().validate_allocation(ptr, 1024);
    EXPECT_TRUE(result);

    cudaFree(ptr);
}

TEST_F(MemorySafetyTest, CheckUninitializedDetectsPoison) {
    std::vector<uint8_t> data(256);
    std::fill(data.begin(), data.end(), 0xFE);

    void* d_ptr = nullptr;
    cudaMalloc(&d_ptr, 256);
    cudaMemcpy(d_ptr, data.data(), 256, cudaMemcpyHostToDevice);

    auto result = MemorySafetyValidator::instance().check_uninitialized(d_ptr, 256);
    EXPECT_FALSE(result);

    cudaFree(d_ptr);
}

TEST_F(MemorySafetyTest, EnableDisable) {
    MemorySafetyValidator::instance().disable();
    EXPECT_FALSE(MemorySafetyValidator::instance().is_enabled());

    MemorySafetyValidator::instance().enable();
    EXPECT_TRUE(MemorySafetyValidator::instance().is_enabled());
}

TEST_F(MemorySafetyTest, ValidationCount) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, 100);

    MemorySafetyValidator::instance().validate_allocation(ptr, 100);

    EXPECT_EQ(MemorySafetyValidator::instance().validation_count(), 1);

    cudaFree(ptr);
}

TEST_F(MemorySafetyTest, SetTool) {
    MemorySafetyValidator::instance().set_tool(MemorySafetyTool::ComputeSanitizer);
    EXPECT_EQ(MemorySafetyValidator::instance().tool(), MemorySafetyTool::ComputeSanitizer);

    MemorySafetyValidator::instance().set_tool(MemorySafetyTool::PoisonPattern);
    EXPECT_EQ(MemorySafetyValidator::instance().tool(), MemorySafetyTool::PoisonPattern);
}

}  // namespace cuda::testing::test
