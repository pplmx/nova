/**
 * @file validate_csr_null_test.cpp
 * @brief validate_csr must report an unfilled graph, not dereference its null arrays
 *
 * CSRGraph's default constructor leaves row_offsets, columns and weights null. validate_csr
 * began with
 *
 *     if (graph.row_offsets[0] != 0) return false;
 *
 * so validating a graph that was constructed but never filled in read through a null pointer.
 * The function exists to answer "is this graph well formed", and a graph holding no arrays is
 * not well formed — the answer is false, not a crash.
 *
 * The existing csr_graph_test cases only ever assert validate_csr on uploaded graphs, so all of
 * them return true and none of them reached this.
 */

#include <cuda/graph/csr_graph.h>

#include <gtest/gtest.h>

namespace validate_csr_null_test {

    TEST(ValidateCsrNullTest, ADefaultConstructedGraphIsInvalidNotFatal) {
        cuda::graph::CSRGraph graph;  // all arrays null, counts zero

        // Before the guard this dereferenced row_offsets[0] on a null pointer.
        EXPECT_FALSE(cuda::graph::validate_csr(graph)) << "a graph with no arrays is not a valid CSR structure";
    }

    TEST(ValidateCsrNullTest, AClearedGraphIsInvalidNotFatal) {
        cuda::graph::CSRGraph graph(3, 2);
        graph.clear();

        // clear() releases the arrays and nulls them, and resets the counts. Validating afterwards
        // is the same situation as above and must give the same answer.
        EXPECT_FALSE(cuda::graph::validate_csr(graph));
    }

    TEST(ValidateCsrNullTest, AnUploadedGraphIsStillValid) {
        // The guard must not make valid graphs fail: this is the case the existing suite covers,
        // kept here so a regression in the opposite direction shows up next to the new checks.
        cuda::graph::CSRGraph graph(3, 2);
        graph.row_offsets[0] = 0;
        graph.row_offsets[1] = 1;
        graph.row_offsets[2] = 2;
        graph.row_offsets[3] = 2;
        graph.columns[0] = 1;
        graph.columns[1] = 2;

        EXPECT_TRUE(cuda::graph::validate_csr(graph));
    }

}  // namespace validate_csr_null_test
