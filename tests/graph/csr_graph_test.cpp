#include <gtest/gtest.h>
#include <vector>
#include "cuda/graph/csr_graph.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

using cuda::memory::Buffer;
using namespace cuda::graph;

namespace {

// Directed graph that distinguishes forward SpMV from its transpose:
//   0 -> {1},  1 -> {2},  2 -> {},  3 -> {0, 1}
// Forward  A*x:  y0=x1, y1=x2, y2=0, y3=x0+x1
// Transpose:     y0=x3, y1=x0+x3, y2=x1, y3=0
std::vector<std::vector<int>> directed_graph() {
    return {
        {1},
        {2},
        {},
        {0, 1}
    };
}

}  // namespace

class CSRGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess && err != cudaErrorNoDevice) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        cudaGetLastError();
    }

    std::vector<std::vector<int>> create_simple_graph() {
        return {
            {1, 2},
            {0, 2},
            {0, 1},
            {}
        };
    }
};

TEST_F(CSRGraphTest, CreateFromAdjacency) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);

    EXPECT_EQ(graph->vertices(), 4);
    EXPECT_GE(graph->edges(), 5);
}

TEST_F(CSRGraphTest, CreateFromEdges) {
    // Edges: 0->1, 0->2, 1->2, 3->0 (out of order, repeated source).
    std::vector<int> src = {3, 0, 1, 0};
    std::vector<int> dst = {0, 2, 2, 1};
    std::vector<float> w = {7.0f, 5.0f, 3.0f, 2.0f};

    auto graph = create_csr_from_edges(src.data(), dst.data(), w.data(), 4, 4);
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->vertices(), 4);
    EXPECT_EQ(graph->edges(), 4);

    // Rows hold each source's out-neighbors (in insertion order per source).
    EXPECT_EQ(graph->degree(0), 2);
    EXPECT_EQ(graph->degree(1), 1);
    EXPECT_EQ(graph->degree(2), 0);
    EXPECT_EQ(graph->degree(3), 1);
    EXPECT_EQ(graph->row_offsets[0], 0);
    EXPECT_EQ(graph->row_offsets[4], 4);
    EXPECT_TRUE(validate_csr(*graph));
}

TEST_F(CSRGraphTest, RowOffsetsAreValid) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);

    EXPECT_EQ(graph->row_offsets[0], 0);
    EXPECT_LT(graph->row_offsets[1], graph->row_offsets[4]);
    EXPECT_EQ(graph->row_offsets[4], graph->edges());
}

TEST_F(CSRGraphTest, DegreeCalculation) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);

    EXPECT_EQ(graph->degree(0), 2);
    EXPECT_EQ(graph->degree(1), 2);
    EXPECT_EQ(graph->degree(2), 2);
    EXPECT_EQ(graph->degree(3), 0);
}

TEST_F(CSRGraphTest, DefaultConstruction) {
    CSRGraph graph;
    EXPECT_EQ(graph.vertices(), 0);
    EXPECT_EQ(graph.edges(), 0);
}

TEST_F(CSRGraphTest, ParameterizedConstruction) {
    CSRGraph graph(10, 20);
    EXPECT_EQ(graph.vertices(), 10);
    EXPECT_EQ(graph.edges(), 20);
}

TEST_F(CSRGraphTest, MemoryUsageIsPositive) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);

    size_t mem = graph->memory_usage();
    EXPECT_GT(mem, 0);
}

TEST_F(CSRGraphTest, ValidateSmallGraph) {
    std::vector<std::vector<int>> adj = {{1}, {0}};
    auto graph = create_csr_from_adjacency(adj);
    EXPECT_TRUE(validate_csr(*graph));
}

TEST_F(CSRGraphTest, ClearFreesMemory) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);
    graph->clear();

    EXPECT_EQ(graph->row_offsets, nullptr);
    EXPECT_EQ(graph->columns, nullptr);
    EXPECT_EQ(graph->weights, nullptr);
}

TEST_F(CSRGraphTest, SingleVertexNoEdges) {
    std::vector<std::vector<int>> adj = {{}};
    auto graph = create_csr_from_adjacency(adj);

    EXPECT_EQ(graph->vertices(), 1);
    EXPECT_EQ(graph->edges(), 0);
    EXPECT_EQ(graph->degree(0), 0);
}

TEST_F(CSRGraphTest, LinearChainGraph) {
    std::vector<std::vector<int>> adj = {
        {1},
        {0, 2},
        {1, 3},
        {2}
    };
    auto graph = create_csr_from_adjacency(adj);

    EXPECT_EQ(graph->vertices(), 4);
    EXPECT_EQ(graph->edges(), 6);
    EXPECT_EQ(graph->degree(0), 1);
    EXPECT_EQ(graph->degree(1), 2);
    EXPECT_EQ(graph->degree(2), 2);
    EXPECT_EQ(graph->degree(3), 1);
}

TEST_F(CSRGraphTest, CsrMvMatchesReference) {
    auto adj = directed_graph();
    auto graph = create_csr_from_adjacency(adj);
    graph->upload();

    std::vector<float> x = {1.0f, 2.0f, 4.0f, 8.0f};
    Buffer<float> d_x(x.size());
    Buffer<float> d_y(4);
    d_x.copy_from(x.data(), x.size());

    csr_mv(*graph, d_x.data(), d_y.data());

    std::vector<float> y(4);
    d_y.copy_to(y.data(), 4);
    // Forward A*x: y0=x1, y1=x2, y2=0, y3=x0+x1
    EXPECT_FLOAT_EQ(y[0], 2.0f);
    EXPECT_FLOAT_EQ(y[1], 4.0f);
    EXPECT_FLOAT_EQ(y[2], 0.0f);
    EXPECT_FLOAT_EQ(y[3], 3.0f);
}

TEST_F(CSRGraphTest, CsrMvTransposeMatchesReference) {
    auto adj = directed_graph();
    auto graph = create_csr_from_adjacency(adj);
    graph->upload();

    std::vector<float> x = {1.0f, 2.0f, 4.0f, 8.0f};
    Buffer<float> d_x(x.size());
    Buffer<float> d_y(4);
    d_x.copy_from(x.data(), x.size());

    csr_mv_transpose(*graph, d_x.data(), d_y.data());

    std::vector<float> y(4);
    d_y.copy_to(y.data(), 4);
    // Transpose: y0=x3, y1=x0+x3, y2=x1, y3=0
    EXPECT_FLOAT_EQ(y[0], 8.0f);
    EXPECT_FLOAT_EQ(y[1], 9.0f);
    EXPECT_FLOAT_EQ(y[2], 2.0f);
    EXPECT_FLOAT_EQ(y[3], 0.0f);
}

TEST_F(CSRGraphTest, ComputeDegreesMatchesHostDegrees) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);
    graph->upload();

    Buffer<int> d_degrees(4);
    compute_degrees(*graph, d_degrees.data());

    std::vector<int> degrees(4);
    d_degrees.copy_to(degrees.data(), 4);
    for (int v = 0; v < 4; ++v) {
        EXPECT_EQ(degrees[v], graph->degree(v));
    }
}

TEST_F(CSRGraphTest, ComputeInDegreesPopulatesBuffer) {
    // Directed graph 0->1, 1->2, 2->{}, 3->{0,1}: in-degrees {1, 2, 1, 0}.
    auto adj = directed_graph();
    auto graph = create_csr_from_adjacency(adj);
    graph->upload();

    Buffer<int> d_out(4);
    Buffer<int> d_in(4);
    compute_degrees(*graph, d_out.data(), d_in.data());

    std::vector<int> in(4);
    d_in.copy_to(in.data(), 4);
    std::vector<int> expected_in = {1, 2, 1, 0};
    for (int v = 0; v < 4; ++v) {
        EXPECT_EQ(in[v], expected_in[v]);
    }
}
