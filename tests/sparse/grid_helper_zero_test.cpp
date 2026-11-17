/**
 * @file grid_helper_zero_test.cpp
 * @brief The calc_grid_* helpers must not return a zero-block grid
 *
 * All four computed `(n + block - 1) / block`, which is 0 when n is 0. A launch with a zero
 * dimension is rejected by CUDA with "invalid configuration argument", so a caller that
 * legitimately passes an empty batch or a graph with no vertices got a driver error where the
 * intended behaviour is a launch that does nothing.
 *
 * The same gap was fixed in csr_mv and csr_mv_transpose and in pagerank; these are the generic
 * helpers those callers use, so every new call site inherited it.
 *
 * No existing test called calc_grid_1d, _2d or _3d — the only calc_grid coverage was for
 * calc_grid_auto, and it used non-zero sizes.
 */

#include <cuda/algo/kernel_launcher.h>

#include <gtest/gtest.h>

namespace grid_helper_zero_test {

TEST(GridHelperZeroTest, OneDimensionalZeroSizeStillLaunchesOneBlock) {
    const dim3 grid = cuda::detail::calc_grid_1d(0);
    EXPECT_EQ(grid.x, 1u)
        << "a zero-size input must not produce a zero-block grid";
    EXPECT_EQ(grid.y, 1u);
    EXPECT_EQ(grid.z, 1u);
}

TEST(GridHelperZeroTest, OneDimensionalSizingIsOtherwiseUnchanged) {
    // 256 threads per block by default: 1 element is one block, 256 is one, 257 is two.
    EXPECT_EQ(cuda::detail::calc_grid_1d(1).x, 1u);
    EXPECT_EQ(cuda::detail::calc_grid_1d(256).x, 1u);
    EXPECT_EQ(cuda::detail::calc_grid_1d(257).x, 2u);
    EXPECT_EQ(cuda::detail::calc_grid_1d(1024).x, 4u);
}

TEST(GridHelperZeroTest, TwoDimensionalZeroOnEitherAxisStillLaunches) {
    const dim3 both_zero = cuda::detail::calc_grid_2d(0, 0);
    EXPECT_EQ(both_zero.x, 1u);
    EXPECT_EQ(both_zero.y, 1u);

    const dim3 width_zero = cuda::detail::calc_grid_2d(0, 32);
    EXPECT_EQ(width_zero.x, 1u) << "the zero axis is clamped";
    EXPECT_GT(width_zero.y, 0u);

    const dim3 height_zero = cuda::detail::calc_grid_2d(32, 0);
    EXPECT_GT(height_zero.x, 0u);
    EXPECT_EQ(height_zero.y, 1u) << "the zero axis is clamped";
}

TEST(GridHelperZeroTest, ThreeDimensionalZeroOnAnyAxisStillLaunches) {
    const dim3 all_zero = cuda::detail::calc_grid_3d(0, 0, 0);
    EXPECT_EQ(all_zero.x, 1u);
    EXPECT_EQ(all_zero.y, 1u);
    EXPECT_EQ(all_zero.z, 1u);

    const dim3 z_zero = cuda::detail::calc_grid_3d(8, 8, 0);
    EXPECT_GT(z_zero.x, 0u);
    EXPECT_GT(z_zero.y, 0u);
    EXPECT_EQ(z_zero.z, 1u);
}

TEST(GridHelperZeroTest, TwoAndThreeDimensionalSizingIsOtherwiseUnchanged) {
    // 16x16 blocks by default for 2D.
    EXPECT_EQ(cuda::detail::calc_grid_2d(16, 16).x, 1u);
    EXPECT_EQ(cuda::detail::calc_grid_2d(32, 16).x, 2u);
    EXPECT_EQ(cuda::detail::calc_grid_2d(16, 32).y, 2u);

    // 8x8x8 blocks by default for 3D.
    const dim3 grid = cuda::detail::calc_grid_3d(8, 16, 24);
    EXPECT_EQ(grid.x, 1u);
    EXPECT_EQ(grid.y, 2u);
    EXPECT_EQ(grid.z, 3u);
}

}  // namespace grid_helper_zero_test
