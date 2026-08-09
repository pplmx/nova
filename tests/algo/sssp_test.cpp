#include <algorithm>
#include <gtest/gtest.h>

#include "cuda/algo/sssp.h"
#include "cuda/graph/csr_graph.h"

namespace cuda::algo::sssp::test {

class SSSPTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        cudaDeviceSynchronize();
    }
    void TearDown() override {}
};

TEST_F(SSSPTest, ConfigSetters) {
    SSSPConfig config;
    config.default_delta = 2.0f;
    config.max_iterations = 500;

    cuda::algo::sssp::set_config(config);

    auto retrieved = cuda::algo::sssp::get_config();
    EXPECT_EQ(retrieved.default_delta, 2.0f);
    EXPECT_EQ(retrieved.max_iterations, 500);
}

TEST_F(SSSPTest, INFConstant) {
    EXPECT_TRUE(cuda::algo::sssp::INF > 1e20f);
    EXPECT_FALSE(cuda::algo::sssp::INF < cuda::algo::sssp::INF);
}

TEST_F(SSSPTest, BellmanFordSimple) {
    constexpr int num_vertices = 4;
    constexpr int num_edges = 5;

    std::vector<float> values = {1.0f, 4.0f, 2.0f, 5.0f, 1.0f};
    std::vector<int> row_offsets = {0, 2, 4, 5, 5};
    std::vector<int> col_indices = {1, 2, 0, 3, 2};

    graph::CSRGraph graph(num_vertices, num_edges);
    std::copy(row_offsets.begin(), row_offsets.end(), graph.row_offsets);
    std::copy(col_indices.begin(), col_indices.end(), graph.columns);
    std::copy(values.begin(), values.end(), graph.weights);

    cuda::memory::Buffer<float> distances(num_vertices);

    cuda::algo::sssp::bellman_ford(graph, 0, distances.data());

    std::vector<float> result(num_vertices);
    cudaMemcpy(result.data(), distances.data(), num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[0], 0.0f);
    EXPECT_EQ(result[1], 1.0f);
    EXPECT_LE(result[2], 3.0f);
}

TEST_F(SSSPTest, DeltaSteppingSimple) {
    constexpr int num_vertices = 3;
    constexpr int num_edges = 3;

    std::vector<float> values = {1.0f, 1.0f, 1.0f};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {1, 2, 0};

    graph::CSRGraph graph(num_vertices, num_edges);
    std::copy(row_offsets.begin(), row_offsets.end(), graph.row_offsets);
    std::copy(col_indices.begin(), col_indices.end(), graph.columns);
    std::copy(values.begin(), values.end(), graph.weights);

    cuda::memory::Buffer<float> distances(num_vertices);

    cuda::algo::sssp::delta_stepping(graph, 0, distances.data(), 1.0f);

    std::vector<float> result(num_vertices);
    cudaMemcpy(result.data(), distances.data(), num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[0], 0.0f);
    EXPECT_EQ(result[1], 1.0f);
    EXPECT_EQ(result[2], 1.0f);
}

TEST_F(SSSPTest, DeltaSteppingLongerPaths) {
    constexpr int num_vertices = 6;
    constexpr int num_edges = 6;

    std::vector<float> values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<int> row_offsets = {0, 1, 2, 3, 4, 5, 6};
    std::vector<int> col_indices = {1, 2, 0, 3, 2, 4};

    graph::CSRGraph graph(num_vertices, num_edges);
    std::copy(row_offsets.begin(), row_offsets.end(), graph.row_offsets);
    std::copy(col_indices.begin(), col_indices.end(), graph.columns);
    std::copy(values.begin(), values.end(), graph.weights);

    auto distances = cuda::algo::sssp::compute_distances<float>(graph, 0, 1.0f);

    std::vector<float> result(num_vertices);
    cudaMemcpy(result.data(), distances.data(), num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[0], 0.0f);
    EXPECT_EQ(result[1], 5.0f);
}

TEST_F(SSSPTest, ComputeDistances) {
    constexpr int num_vertices = 2;
    constexpr int num_edges = 1;

    std::vector<float> values = {5.0f};
    std::vector<int> row_offsets = {0, 1, 1};
    std::vector<int> col_indices = {1};

    graph::CSRGraph graph(num_vertices, num_edges);
    std::copy(row_offsets.begin(), row_offsets.end(), graph.row_offsets);
    std::copy(col_indices.begin(), col_indices.end(), graph.columns);
    std::copy(values.begin(), values.end(), graph.weights);

    auto distances = cuda::algo::sssp::compute_distances<float>(graph, 0, 1.0f);

    std::vector<float> result(num_vertices);
    cudaMemcpy(result.data(), distances.data(), num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[0], 0.0f);
    EXPECT_EQ(result[1], 5.0f);
}

}  // namespace cuda::algo::sssp::test
