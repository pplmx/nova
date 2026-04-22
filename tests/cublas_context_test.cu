#include <gtest/gtest.h>
#include "cuda/device/cublas_context.h"
#include "cuda/memory/buffer.h"
#include <vector>

using cuda::memory::Buffer;

class CublasContextTest : public ::testing::Test {};

TEST_F(CublasContextTest, ContextCreation) {
    cuda::device::CublasContext ctx;
    EXPECT_NE(ctx.get(), nullptr);
}

TEST_F(CublasContextTest, MoveConstruction) {
    cuda::device::CublasContext ctx1;
    auto handle = ctx1.get();

    cuda::device::CublasContext ctx2(std::move(ctx1));
    EXPECT_EQ(ctx2.get(), handle);
}

TEST_F(CublasContextTest, MoveAssignment) {
    cuda::device::CublasContext ctx1;
    cuda::device::CublasContext ctx2;
    auto handle1 = ctx1.get();

    ctx2 = std::move(ctx1);
    EXPECT_EQ(ctx2.get(), handle1);
}

TEST_F(CublasContextTest, CanBeUsedInSubsequentOperations) {
    cuda::device::CublasContext ctx1;
    cuda::device::CublasContext ctx2;
    EXPECT_NE(ctx1.get(), nullptr);
    EXPECT_NE(ctx2.get(), nullptr);
}
