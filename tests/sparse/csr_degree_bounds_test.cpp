/**
 * @file csr_degree_bounds_test.cpp
 * @brief CSRGraph::degree indexes row_offsets and must reject out-of-range vertices
 *
 * degree(v) returns row_offsets[v + 1] - row_offsets[v]. It is the only accessor in the class
 * that indexes an array — vertices() and edges() just read a count — and it checked nothing:
 * an out-of-range v read past the array, and a default-constructed graph dereferenced null.
 *
 * The existing cases in csr_graph_test only ask for legal vertices, so none of them reached
 * either state.
 */

#include <cuda/graph/csr_graph.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace csr_degree_bounds_test {

namespace {

cuda::graph::CSRGraph make_three_vertex_chain() {
    // 0 -> 1 -> 2
    cuda::graph::CSRGraph graph(3, 2);
    graph.row_offsets[0] = 0;
    graph.row_offsets[1] = 1;
    graph.row_offsets[2] = 2;
    graph.row_offsets[3] = 2;
    graph.columns[0] = 1;
    graph.columns[1] = 2;
    return graph;
}

}  // namespace

TEST(CsrDegreeBoundsTest, ALegalVertexReportsItsOutDegree) {
    auto graph = make_three_vertex_chain();
    EXPECT_EQ(graph.degree(0), 1);
    EXPECT_EQ(graph.degree(1), 1);
    EXPECT_EQ(graph.degree(2), 0);
}

TEST(CsrDegreeBoundsTest, AVertexPastTheEndIsRejected) {
    auto graph = make_three_vertex_chain();
    // v = 3 would read row_offsets[4] on a 4-element array — one past the end.
    EXPECT_THROW(graph.degree(3), std::out_of_range);
    EXPECT_THROW(graph.degree(99), std::out_of_range);
}

TEST(CsrDegreeBoundsTest, ANegativeVertexIsRejected) {
    auto graph = make_three_vertex_chain();
    // v = -1 reads row_offsets[-1].
    EXPECT_THROW(graph.degree(-1), std::out_of_range);
}

TEST(CsrDegreeBoundsTest, ADefaultConstructedGraphIsRejectedNotFatal) {
    cuda::graph::CSRGraph graph;  // row_offsets is null
    // Before the guard this dereferenced null for any v, including 0.
    EXPECT_THROW(graph.degree(0), std::out_of_range);
}

TEST(CsrDegreeBoundsTest, AClearedGraphIsRejectedNotFatal) {
    auto graph = make_three_vertex_chain();
    graph.clear();
    // clear() releases the arrays and nulls them, so this is the same state as above.
    EXPECT_THROW(graph.degree(0), std::out_of_range);
}

}  // namespace csr_degree_bounds_test
