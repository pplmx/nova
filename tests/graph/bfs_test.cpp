#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include "cuda/graph/csr_graph.h"
#include "cuda/graph/bfs.h"

using namespace cuda::graph;

class BFSTest : public ::testing::Test {
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

TEST_F(BFSTest, BFSResultIsMovable) {
    BFSResult a(4);
    a.init_source(0);

    // Move construction must transfer ownership and empty the source. With the
    // old implicit shallow copy, `a` would keep dangling pointers and the two
    // objects would double-free.
    BFSResult b = std::move(a);
    EXPECT_EQ(b.num_vertices, 4);
    EXPECT_EQ(b.distance_to(0), 0);
    EXPECT_EQ(a.num_vertices, 0);
    EXPECT_EQ(a.distances, nullptr);
    EXPECT_EQ(a.d_distances, nullptr);
    EXPECT_EQ(a.visited, nullptr);

    // Move assignment re-seats an existing object without leaking.
    BFSResult c(4);
    c = std::move(b);
    EXPECT_EQ(c.num_vertices, 4);
    EXPECT_EQ(c.distance_to(0), 0);
    EXPECT_EQ(b.distances, nullptr);

    c.clear();  // must not double-free
}

TEST_F(BFSTest, BfsAsyncReusesResultWithoutDoubleFree) {
    // Directed chain 0 -> {1,3}, 1 -> 2, 2 -> 3.
    std::vector<std::vector<int>> adj(4);
    adj[0] = {1, 3};
    adj[1] = {2};
    adj[2] = {3};
    auto graph = create_csr_from_adjacency(adj);

    BFSResult result(4);
    bfs_async(*graph, result, 0);
    EXPECT_EQ(result.distance_to(0), 0);
    EXPECT_EQ(result.distance_to(1), 1);
    EXPECT_EQ(result.distance_to(2), 2);
    EXPECT_EQ(result.distance_to(3), 1);
    EXPECT_EQ(result.visited_count, 4);

    // A second call re-seats the result; with implicit shallow copy assignment
    // the first bfs() temporary frees the buffers the result still points at
    // (double free / use-after-free).
    bfs_async(*graph, result, 1);
    EXPECT_EQ(result.num_vertices, 4);
    EXPECT_EQ(result.distance_to(0), -1);
    EXPECT_EQ(result.distance_to(1), 0);
    EXPECT_EQ(result.distance_to(2), 1);
    EXPECT_EQ(result.distance_to(3), 2);
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

TEST_F(BFSTest, BFSFindsShortestPathsAcrossMultipleLevels) {
    // Chain 0 -> 1 -> 2 plus a shortcut 0 -> 3. A correct level-synchronous
    // BFS must reach depth 2 (node 2) through node 1.
    std::vector<std::vector<int>> adj(4);
    adj[0] = {1, 3};
    adj[1] = {2};
    adj[2] = {3};
    auto graph = create_csr_from_adjacency(adj);

    auto result = bfs(*graph, 0);

    EXPECT_EQ(result.visited_count, 4);
    EXPECT_EQ(result.max_distance, 2);
    EXPECT_EQ(result.distance_to(0), 0);
    EXPECT_EQ(result.distance_to(1), 1);
    EXPECT_EQ(result.distance_to(2), 2);
    EXPECT_EQ(result.distance_to(3), 1);
}

TEST_F(BFSTest, BFSReachesDepthThreeThroughLongChain) {
    // 0 -> 1 -> 2 -> 3: every node must be discovered at its own level.
    std::vector<std::vector<int>> adj(6);
    adj[0] = {1};
    adj[1] = {2};
    adj[2] = {3};
    auto graph = create_csr_from_adjacency(adj);

    auto result = bfs(*graph, 0);

    EXPECT_EQ(result.visited_count, 4);
    EXPECT_EQ(result.max_distance, 3);
    EXPECT_EQ(result.distance_to(0), 0);
    EXPECT_EQ(result.distance_to(1), 1);
    EXPECT_EQ(result.distance_to(2), 2);
    EXPECT_EQ(result.distance_to(3), 3);
    EXPECT_EQ(result.distance_to(4), -1);
    EXPECT_EQ(result.distance_to(5), -1);
}

TEST_F(BFSTest, BFSFindsEverythingInCompleteTriangle) {
    // Triangle {0,1,2}: all reachable at distance 1.
    auto adj = create_simple_graph();
    auto graph = create_csr_from_adjacency(adj);

    auto result = bfs(*graph, 0);

    EXPECT_EQ(result.distance_to(0), 0);
    EXPECT_EQ(result.distance_to(1), 1);
    EXPECT_EQ(result.distance_to(2), 1);
    EXPECT_EQ(result.distance_to(3), -1);
}
