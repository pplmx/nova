/**
 * @file csr_from_edges_validation_test.cpp
 * @brief create_csr_from_edges must reject inputs that would corrupt the graph
 *
 * The function builds CSR arrays from caller-supplied edge lists and did no validation:
 *
 *   counts[src_vertices[i]]++          indexes a host array by vertex id with no bound
 *   graph->columns[pos] = dst_vertices[i]   stores a destination with no bound
 *
 * The adjacency overload in the same file validates its inputs and is documented; this one had
 * neither. A source id outside [0, num_vertices) corrupted the heap, and a destination outside
 * it silently produced a graph validate_csr would later call invalid.
 *
 * These cases pin the rejections. A valid call is included so the guards cannot pass by
 * refusing everything.
 */

#include <cuda/graph/csr_graph.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace csr_from_edges_validation_test {

TEST(CsrFromEdgesValidationTest, ANonPositiveVertexCountIsRejected) {
    const std::vector<int> src{0};
    const std::vector<int> dst{1};

    auto call = [&] {
        return cuda::graph::create_csr_from_edges(src.data(), dst.data(), nullptr, 0, 1);
    };
    EXPECT_THROW(call(), std::invalid_argument);
}

TEST(CsrFromEdgesValidationTest, ANegativeEdgeCountIsRejected) {
    const std::vector<int> src{0};
    const std::vector<int> dst{1};

    auto call = [&] {
        return cuda::graph::create_csr_from_edges(src.data(), dst.data(), nullptr, 2, -1);
    };
    EXPECT_THROW(call(), std::invalid_argument);
}

TEST(CsrFromEdgesValidationTest, NullArraysWithEdgesAreRejected) {
    // Before the guard this dereferenced src_vertices[0] on a null pointer.
    auto call = [&] {
        return cuda::graph::create_csr_from_edges(nullptr, nullptr, nullptr, 2, 3);
    };
    EXPECT_THROW(call(), std::invalid_argument);
}

TEST(CsrFromEdgesValidationTest, ASourceOutsideTheVertexRangeIsRejected) {
    // counts[5] on a 4-entry array: heap corruption, not an error, before the guard.
    const std::vector<int> src{0, 5};
    const std::vector<int> dst{1, 2};

    auto call = [&] {
        return cuda::graph::create_csr_from_edges(src.data(), dst.data(), nullptr, 4, 2);
    };
    EXPECT_THROW(call(), std::out_of_range);
}

TEST(CsrFromEdgesValidationTest, ADestinationOutsideTheVertexRangeIsRejected) {
    // Storing 9 in a graph of 4 vertices makes an invalid CSR structure that validate_csr
    // would reject after the fact; the constructor should not produce it in the first place.
    const std::vector<int> src{0};
    const std::vector<int> dst{9};

    auto call = [&] {
        return cuda::graph::create_csr_from_edges(src.data(), dst.data(), nullptr, 4, 1);
    };
    EXPECT_THROW(call(), std::out_of_range);
}

TEST(CsrFromEdgesValidationTest, AValidEdgeListStillBuildsAValidGraph) {
    const std::vector<int> src{0, 0, 1};
    const std::vector<int> dst{1, 2, 2};
    const std::vector<float> weights{1.0F, 2.0F, 3.0F};

    auto graph = cuda::graph::create_csr_from_edges(src.data(), dst.data(),
                                                    weights.data(), 3, 3);
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->vertices(), 3);
    EXPECT_EQ(graph->edges(), 3);
    EXPECT_TRUE(cuda::graph::validate_csr(*graph))
        << "the guards must not reject a well-formed edge list";
}

TEST(CsrFromEdgesValidationTest, NullWeightsMeansUnitWeight) {
    const std::vector<int> src{0};
    const std::vector<int> dst{1};

    auto graph = cuda::graph::create_csr_from_edges(src.data(), dst.data(),
                                                    nullptr, 2, 1);
    ASSERT_NE(graph, nullptr);
    // weights is documented as optional; the branch in the loop supplies 1.0f.
    EXPECT_FLOAT_EQ(graph->weights[0], 1.0F);
}

}  // namespace csr_from_edges_validation_test
