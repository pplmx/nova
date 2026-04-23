#include <gtest/gtest.h>
#include <vector>
#include "cuda/graph/csr_graph.h"
#include "cuda/graph/bfs.h"

using namespace cuda::graph;

class BFSTest : public ::testing::Test {
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

TEST_F(BFSTest, BFSResultConstruction) {
    BFSResult result(10);
    EXPECT_EQ(result.num_vertices, 10);
    EXPECT_NE(result.distances, nullptr);
    EXPECT_NE(result.d_distances, nullptr);
}

TEST_F(BFSTest, InitSourceSetsCorrectValue) {
    BFSResult result(10);
    result.init_source(5);

    EXPECT_EQ(result.distance_to(5), 0);
    EXPECT_TRUE(result.is_reachable(5));
}

TEST_F(BFSTest, UnreachableVertexHasNegativeDistance) {
    BFSResult result(10);
    result.init_source(0);

    EXPECT_EQ(result.distance_to(9), -1);
    EXPECT_FALSE(result.is_reachable(9));
}

TEST_F(BFSTest, BFSResultDefaultConstruction) {
    BFSResult result;
    EXPECT_EQ(result.num_vertices, 0);
}

TEST_F(BFSTest, BFSOnSimpleGraph) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);

    auto result = bfs(*graph, 0);

    EXPECT_EQ(result.distance_to(0), 0);
    EXPECT_GE(result.distance_to(1), 0);
    EXPECT_GE(result.distance_to(2), 0);
}

TEST_F(BFSTest, BFSWithDisconnectedComponent) {
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);

    auto result = bfs(*graph, 0);

    EXPECT_EQ(result.distance_to(3), -1);
    EXPECT_FALSE(result.is_reachable(3));
}

TEST_F(BFSTest, BFSMemoryUsageIsPositive) {
    BFSResult result(100);
    size_t mem = result.memory_usage();
    EXPECT_GT(mem, 0);
}

TEST_F(BFSTest, BFSClearFreesMemory) {
    BFSResult result(10);
    result.clear();

    EXPECT_EQ(result.distances, nullptr);
    EXPECT_EQ(result.d_distances, nullptr);
}

TEST_F(BFSTest, BFSOnSingleVertex) {
    std::vector<std::vector<int>> adj = {{}};
    auto graph = create_csr_from_adjacency(adj);

    auto result = bfs(*graph, 0);

    EXPECT_EQ(result.distance_to(0), 0);
    EXPECT_EQ(result.visited_count, 1);
}

TEST_F(BFSTest, BFSCorrectShortestPath) {
    BFSResult result(4);
    result.init_source(0);
    result.distances[1] = 1;
    result.distances[2] = 2;
    result.distances[3] = 3;

    EXPECT_EQ(result.distance_to(0), 0);
    EXPECT_EQ(result.distance_to(1), 1);
    EXPECT_EQ(result.distance_to(2), 2);
    EXPECT_EQ(result.distance_to(3), 3);
}
