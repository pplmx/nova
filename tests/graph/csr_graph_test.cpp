#include <gtest/gtest.h>
#include <vector>
#include "cuda/graph/csr_graph.h"

using namespace cuda::graph;

class CSRGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
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
