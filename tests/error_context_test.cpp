#include <gtest/gtest.h>
#include "cuda/device/error.h"

class ErrorContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(ErrorContextTest, OperationContextStoresAllFields) {
    cuda::device::OperationContext ctx;
    ctx.operation_name = "test_op";
    ctx.dimensions = size_t{1024};
    ctx.device_id = 0;
    ctx.extra = "test_extra";

    EXPECT_STREQ(ctx.operation_name, "test_op");
    EXPECT_EQ(std::get<size_t>(ctx.dimensions), 1024);
    EXPECT_EQ(ctx.device_id, 0);
    EXPECT_STREQ(ctx.extra.c_str(), "test_extra");
}

TEST_F(ErrorContextTest, OperationContextWithPairDimensions) {
    cuda::device::OperationContext ctx;
    ctx.operation_name = "matrix_mult";
    ctx.dimensions = std::pair<size_t, size_t>{1024, 2048};
    ctx.device_id = 0;

    auto dims = std::get<std::pair<size_t, size_t>>(ctx.dimensions);
    EXPECT_EQ(dims.first, 1024);
    EXPECT_EQ(dims.second, 2048);
}

TEST_F(ErrorContextTest, CudaExceptionWithContextDerivesFromCudaException) {
    auto ctx = cuda::device::OperationContext{
        .operation_name = "test_op",
        .dimensions = size_t{100},
        .device_id = 0
    };

    cuda::device::CudaExceptionWithContext ex(
        cudaErrorInvalidValue, "test_file.cpp", 42, ctx);

    EXPECT_EQ(ex.error(), cudaErrorInvalidValue);
    EXPECT_EQ(ex.context().operation_name, ctx.operation_name);
}

TEST_F(ErrorContextTest, CUDA_CONTEXTMacroCreatesCorrectContext) {
    int device = 0;
    auto ctx = CUDA_CONTEXT(test_operation, size_t{512}, device);

    EXPECT_STREQ(ctx.operation_name, "test_operation");
    EXPECT_EQ(std::get<size_t>(ctx.dimensions), 512);
    EXPECT_EQ(ctx.device_id, 0);
}

TEST_F(ErrorContextTest, CUDA_VALIDATE_SIZEThrowsOnInvalidSize) {
    EXPECT_THROW({
        CUDA_VALIDATE_SIZE(1000, 500, test_operation);
    }, cuda::device::CudaExceptionWithContext);
}

TEST_F(ErrorContextTest, CUDA_VALIDATE_SIZEPassesOnValidSize) {
    EXPECT_NO_THROW({
        CUDA_VALIDATE_SIZE(100, 500, test_operation);
    });
}

TEST_F(ErrorContextTest, CudaExceptionWithContextIsCatchableAsRuntimeError) {
    try {
        CUDA_VALIDATE_SIZE(1000, 500, test_operation);
        FAIL() << "Expected exception";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("CUDA") != std::string::npos ||
                    msg.find("test_operation") != std::string::npos);
    }
}
