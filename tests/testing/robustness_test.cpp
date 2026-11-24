#include <gtest/gtest.h>

#include "cuda/testing/test_isolation.h"
#include "cuda/testing/layer_error_injection.h"
#include "cuda/testing/boundary_testing.h"
#include "cuda/testing/fp_determinism.h"

namespace cuda::testing::test {

    class TestIsolationTest : public ::testing::Test {};

    TEST_F(TestIsolationTest, CreateContext) {
        auto context = TestIsolationContext::create();
        EXPECT_TRUE(context != nullptr);
        EXPECT_TRUE(context->is_isolated());
    }

    TEST_F(TestIsolationTest, ExecuteIsolated) {
        int counter = 0;
        auto context = TestIsolationContext::create();
        context->execute_isolated([&counter]() { counter = 42; });
        EXPECT_EQ(counter, 42);
    }

    class LayerErrorInjectionTest : public ::testing::Test {};

    TEST_F(LayerErrorInjectionTest, InjectAtLayer) {
        auto& injector = LayerAwareErrorInjector::instance();
        injector.reset_all();

        injector.inject_at_layer(LayerBoundary::Memory, cuda::production::ErrorTarget::Allocation, cudaErrorMemoryAllocation);

        EXPECT_TRUE(injector.should_inject_at_layer(LayerBoundary::Memory, cuda::production::ErrorTarget::Allocation));
    }

    TEST_F(LayerErrorInjectionTest, EnableDisableLayer) {
        auto& injector = LayerAwareErrorInjector::instance();

        injector.disable_layer(LayerBoundary::Algorithm);
        EXPECT_FALSE(injector.should_inject_at_layer(LayerBoundary::Algorithm, cuda::production::ErrorTarget::Launch));

        injector.enable_layer(LayerBoundary::Algorithm);
    }

    TEST_F(LayerErrorInjectionTest, LayerNames) {
        EXPECT_STREQ(layer_boundary_name(LayerBoundary::Memory), "Memory");
        EXPECT_STREQ(layer_boundary_name(LayerBoundary::Device), "Device");
        EXPECT_STREQ(layer_boundary_name(LayerBoundary::Algorithm), "Algorithm");
    }

    class BoundaryTest : public ::testing::Test {};

    TEST_F(BoundaryTest, IsWarpAligned) {
        EXPECT_TRUE(is_warp_aligned(32));
        EXPECT_TRUE(is_warp_aligned(64));
        EXPECT_TRUE(is_warp_aligned(256));
        EXPECT_FALSE(is_warp_aligned(33));
        EXPECT_FALSE(is_warp_aligned(65));
    }

    TEST_F(BoundaryTest, IsMemoryAligned) {
        void* aligned = reinterpret_cast<void*>(0x1000);
        void* misaligned = reinterpret_cast<void*>(0x1001);

        EXPECT_TRUE(is_memory_aligned(aligned));
        EXPECT_FALSE(is_memory_aligned(misaligned));
    }

    TEST_F(BoundaryTest, IsValidBlockSize) {
        EXPECT_TRUE(is_valid_block_size(dim3(256, 1, 1)));
        EXPECT_TRUE(is_valid_block_size(dim3(16, 16, 1)));
        EXPECT_FALSE(is_valid_block_size(dim3(1024, 2, 1)));
        EXPECT_FALSE(is_valid_block_size(dim3(0, 1, 1)));
    }

    class FPDeterminismTest : public ::testing::Test {
    protected:
        void SetUp() override { FPDeterminismControl::instance().reset(); }
    };

    TEST_F(FPDeterminismTest, SetLevel) {
        FPDeterminismControl::instance().set_level(DeterminismLevel::RunToRun);
        EXPECT_EQ(FPDeterminismControl::instance().level(), DeterminismLevel::RunToRun);
    }

    TEST_F(FPDeterminismTest, LevelNames) {
        FPDeterminismControl::instance().set_level(DeterminismLevel::GpuToGpu);
        EXPECT_STREQ(FPDeterminismControl::instance().level_name(), "gpu_to_gpu");
    }

    TEST_F(FPDeterminismTest, EnableDisableFTZ) {
        FPDeterminismControl::instance().enable_flush_to_zero();
        EXPECT_TRUE(FPDeterminismControl::instance().is_flush_to_zero_enabled());

        FPDeterminismControl::instance().disable_flush_to_zero();
        EXPECT_FALSE(FPDeterminismControl::instance().is_flush_to_zero_enabled());
    }

    TEST_F(FPDeterminismTest, ScopedDeterminism) {
        FPDeterminismControl::instance().set_level(DeterminismLevel::NotGuaranteed);

        {
            ScopedDeterminism scope(DeterminismLevel::GpuToGpu);
            EXPECT_EQ(FPDeterminismControl::instance().level(), DeterminismLevel::GpuToGpu);
        }

        EXPECT_EQ(FPDeterminismControl::instance().level(), DeterminismLevel::NotGuaranteed);
    }

    // The bounds checks in layer_error_injection.cpp used a literal 6 in eleven places, decoupled
    // from LayerBoundary and from the injector array's size. Adding a seventh layer would have left
    // them silently ignoring it -- a new layer whose injector could never fire, with no error to
    // show for it. They are derived from the enum now, with a static_assert that fails the build if
    // the two disagree.
    //
    // This case covers the runtime half that survives the change: every valid layer is addressable
    // and an out-of-range value is refused rather than reading past the array.
    TEST(LayerBoundaryRangeTest, EveryValidLayerWorksAndOutOfRangeIsRefused) {
        constexpr size_t kCount = static_cast<size_t>(LayerBoundary::Production) + 1;
        EXPECT_EQ(kCount, 6u) << "LayerBoundary gained a value: this case and the .cpp both need updating";

        for (size_t i = 0; i < kCount; ++i) {
            const auto layer = static_cast<LayerBoundary>(i);
            EXPECT_STRNE(layer_boundary_name(layer), "Unknown") << "layer " << i << " has no name";
            // Round trip through the parser.
            EXPECT_EQ(parse_layer_boundary(layer_boundary_name(layer)), layer);
        }

        // Past the enum: the name must say Unknown rather than indexing LAYER_NAMES out of bounds.
        const auto past_the_end = static_cast<LayerBoundary>(kCount);
        EXPECT_STREQ(layer_boundary_name(past_the_end), "Unknown");
    }

}  // namespace cuda::testing::test
