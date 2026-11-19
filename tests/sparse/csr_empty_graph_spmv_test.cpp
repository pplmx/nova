/**
 * @file csr_empty_graph_spmv_test.cpp
 * @brief csr_mv and csr_mv_transpose must not launch a zero-block grid
 *
 * Both compute their grid from a count that can be zero:
 *
 *     csr_mv            grid = (num_vertices + 255) / 256
 *     csr_mv_transpose  grid = (num_edges    + 255) / 256
 *
 * With the count at zero the grid is zero blocks, which CUDA rejects with "invalid
 * configuration argument". The maths over no rows or no edges is an empty result, so returning
 * without launching is the correct answer rather than an error.
 *
 * compute_degrees already clamps its edge grid with std::max(1, ...); these two had nothing.
 */

#include <cuda/graph/csr_graph.h>
#include <cuda/memory/buffer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>

namespace csr_empty_graph_spmv_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

    }  // namespace

    class CsrEmptyGraphSpMvTest : public ::testing::Test {
    protected:
        void TearDown() override { (void)cudaGetLastError(); }
    };

    TEST_F(CsrEmptyGraphSpMvTest, CsrMvOnAZeroVertexGraphDoesNotLaunch) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        cuda::graph::CSRGraph graph;  // no vertices, no edges, arrays null
        ASSERT_EQ(graph.vertices(), 0);

        // Before the guard this launched 0 blocks and the driver rejected it.
        EXPECT_NO_THROW(cuda::graph::csr_mv(graph, nullptr, nullptr)) << "multiplying an empty matrix by a vector is an empty result, not an error";
    }

    TEST_F(CsrEmptyGraphSpMvTest, CsrMvTransposeOnAZeroEdgeGraphDoesNotLaunch) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Vertices but no edges: the transpose grid is keyed on edges and would be zero.
        cuda::graph::CSRGraph graph(3, 0);
        graph.upload();
        cuda::memory::Buffer<float> y(3);

        EXPECT_NO_THROW(cuda::graph::csr_mv_transpose(graph, nullptr, y.data())) << "a transposed product over no edges is zero everywhere";
    }

    TEST_F(CsrEmptyGraphSpMvTest, ANonEmptyGraphStillMultiplies) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // The guards must not disable the ordinary path. Row 0 has one edge to vertex 1 with
        // weight 2, so y[0] = 2 * x[1].
        cuda::graph::CSRGraph graph(2, 1);
        graph.row_offsets[0] = 0;
        graph.row_offsets[1] = 1;
        graph.row_offsets[2] = 1;
        graph.columns[0] = 1;
        graph.weights[0] = 2.0F;
        graph.upload();

        cuda::memory::Buffer<float> x(2);
        cuda::memory::Buffer<float> y(2);
        const float host_x[2] = {1.0F, 5.0F};
        x.copy_from(host_x, 2);

        ASSERT_NO_THROW(cuda::graph::csr_mv(graph, x.data(), y.data()));

        float host_y[2] = {-1.0F, -1.0F};
        y.copy_to(host_y, 2);
        EXPECT_FLOAT_EQ(host_y[0], 10.0F) << "y[0] = weights[0] * x[columns[0]] = 2 * 5";
        EXPECT_FLOAT_EQ(host_y[1], 0.0F) << "vertex 1 has no out-edges";
    }

}  // namespace csr_empty_graph_spmv_test
