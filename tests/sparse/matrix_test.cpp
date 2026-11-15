#include <gtest/gtest.h>
#include <cuda/sparse/matrix.hpp>

#include <stdexcept>

namespace nova::sparse::test {

template<typename T>
bool approx_equal(T a, T b, T tol = T{1e-5}) {
    return std::abs(a - b) < tol;
}

TEST(SparseMatrixTest, DefaultConstruction) {
    SparseMatrix<float> matrix;
    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);
    EXPECT_EQ(matrix.nnz(), 0);
}

TEST(SparseMatrixTest, ConstructionWithDimensions) {
    SparseMatrix<float> matrix(3, 3, 5);
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nnz(), 5);
    EXPECT_NE(matrix.values(), nullptr);
    EXPECT_NE(matrix.row_offsets(), nullptr);
    EXPECT_NE(matrix.col_indices(), nullptr);
}

TEST(SparseMatrixTest, FromDenseTrivial) {
    std::vector<float> dense = {4.0f, 1.0f, 1.0f, 3.0f};
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 2, 2);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 2);
    EXPECT_EQ(matrix->cols(), 2);
    EXPECT_EQ(matrix->nnz(), 4);
}

TEST(SparseMatrixTest, FromDenseSparse) {
    std::vector<float> dense = {
        4.0f, 0.0f, 1.0f,
        0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 2.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 3);
    EXPECT_EQ(matrix->cols(), 3);
    EXPECT_EQ(matrix->nnz(), 4);
}

TEST(SparseMatrixTest, FromDenseWithThreshold) {
    std::vector<float> dense = {
        4.0f, 0.001f, 1.0f,
        0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 2.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.1f);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->nnz(), 4);
}

TEST(SparseMatrixTest, FromDenseAllZeros) {
    std::vector<float> dense(9, 0.0f);
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3);

    EXPECT_FALSE(matrix.has_value());
}

TEST(SparseMatrixTest, FromHostData) {
    std::vector<float> values = {4.0f, 1.0f, 1.0f, 3.0f};
    std::vector<int> row_offsets = {0, 2, 4};
    std::vector<int> col_indices = {0, 1, 0, 1};

    auto matrix = SparseMatrix<float>::FromHostData(values, row_offsets, col_indices, 2, 2);

    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_EQ(matrix.nnz(), 4);
}

TEST(SparseMatrixTest, FromEdgeList) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 3);

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nnz(), 3);
}

TEST(SparseMatrixTest, FromEdgeListWithWeights) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
    std::vector<float> weights = {0.5f, 1.5f, 2.0f};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 3, &weights);

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nnz(), 3);

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    matrix.copy_to_host(h_values, h_row_offsets, h_col_indices);

    EXPECT_FLOAT_EQ(h_values[0], 0.5f);
    EXPECT_FLOAT_EQ(h_values[1], 1.5f);
    EXPECT_FLOAT_EQ(h_values[2], 2.0f);
}

// Regression: FromEdgeList wrote every row offset as `edges.size()`, so all
// rows claimed the whole edge array. rows()/nnz()/values still looked right,
// which is why the structural assertions above never caught it — the matrix
// multiplied wrong. Pin the per-row structure itself.
TEST(SparseMatrixTest, FromEdgeListBuildsPerRowOffsets) {
    // 0 -> 1, 1 -> 2, 2 -> 0: one out-edge per vertex.
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 3);

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    matrix.copy_to_host(h_values, h_row_offsets, h_col_indices);

    EXPECT_EQ(h_row_offsets, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(h_col_indices, (std::vector<int>{1, 2, 0}));
}

TEST(SparseMatrixTest, FromEdgeListGroupsOutEdgesPerRow) {
    // Vertex 0 has two out-edges, vertex 1 none, vertex 2 one.
    std::vector<std::pair<int, int>> edges = {{0, 1}, {0, 2}, {2, 0}};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 3);

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    matrix.copy_to_host(h_values, h_row_offsets, h_col_indices);

    EXPECT_EQ(h_row_offsets, (std::vector<int>{0, 2, 2, 3}));
    EXPECT_EQ(h_col_indices, (std::vector<int>{1, 2, 0}));
    // A vertex with no out-edges must not inherit its neighbours' rows.
    EXPECT_EQ(h_row_offsets[1], h_row_offsets[2]);
}

TEST(SparseMatrixTest, FromEdgeListWeightsFollowTheirEdges) {
    // Weights must stay paired with their own (src, dst), not merely be some
    // permutation of the input.
    std::vector<std::pair<int, int>> edges = {{0, 2}, {1, 0}, {1, 2}};
    std::vector<float> weights = {7.0f, -3.0f, 0.25f};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 3, &weights);

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    matrix.copy_to_host(h_values, h_row_offsets, h_col_indices);

    EXPECT_EQ(h_row_offsets, (std::vector<int>{0, 1, 3, 3}));
    EXPECT_EQ(h_col_indices, (std::vector<int>{2, 0, 2}));
    EXPECT_FLOAT_EQ(h_values[0], 7.0f);   // row 0 -> col 2
    EXPECT_FLOAT_EQ(h_values[1], -3.0f);  // row 1 -> col 0
    EXPECT_FLOAT_EQ(h_values[2], 0.25f);  // row 1 -> col 2
}

TEST(SparseMatrixTest, FromEdgeListReproducesDenseAdjacency) {
    // End-to-end: the CSR must expand back to the adjacency matrix the edge
    // list describes.
    std::vector<std::pair<int, int>> edges = {{0, 0}, {0, 3}, {2, 1}, {3, 2}};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 4);

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    matrix.copy_to_host(h_values, h_row_offsets, h_col_indices);

    std::vector<float> dense(16, 0.0f);
    for (int r = 0; r < 4; ++r) {
        for (int i = h_row_offsets[r]; i < h_row_offsets[r + 1]; ++i) {
            dense[r * 4 + h_col_indices[i]] += h_values[i];
        }
    }

    const std::vector<float> expected = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    EXPECT_EQ(dense, expected);
}

TEST(SparseMatrixTest, FromEdgeListRejectsOutOfRangeVertex) {
    // A destination outside [0, num_vertices) would make spmv read past x.
    std::vector<std::pair<int, int>> edges = {{0, 3}};
    EXPECT_THROW(SparseMatrix<float>::FromEdgeList(edges, 3), std::invalid_argument);

    std::vector<std::pair<int, int>> negative_src = {{-1, 0}};
    EXPECT_THROW(SparseMatrix<float>::FromEdgeList(negative_src, 3), std::invalid_argument);
}

TEST(SparseMatrixTest, FromEdgeListRejectsWeightCountMismatch) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}};
    std::vector<float> too_few = {1.0f};
    EXPECT_THROW(SparseMatrix<float>::FromEdgeList(edges, 3, &too_few),
                 std::invalid_argument);
}

// Regression: copy_to_host copied `num_rows_ + 1` offsets and `nnz()` values
// unconditionally. A default-constructed matrix owns zero-length buffers, so
// the copy ran with a count past the allocation and CUDA rejected it with
// "invalid argument" — merely inspecting an empty matrix threw.
TEST(SparseMatrixTest, CopyToHostOnDefaultConstructedMatrix) {
    SparseMatrix<float> matrix;

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    EXPECT_NO_THROW(matrix.copy_to_host(h_values, h_row_offsets, h_col_indices));

    EXPECT_TRUE(h_values.empty());
    EXPECT_TRUE(h_row_offsets.empty());
    EXPECT_TRUE(h_col_indices.empty());
}

TEST(SparseMatrixTest, CopyToHostOnZeroVertexMatrix) {
    // Built with an explicit row_offsets[0], so the buffer really holds 1 int.
    auto matrix = SparseMatrix<float>::FromHostData({}, {0}, {}, 0, 0);

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    EXPECT_NO_THROW(matrix.copy_to_host(h_values, h_row_offsets, h_col_indices));

    EXPECT_EQ(h_row_offsets, (std::vector<int>{0}));
    EXPECT_TRUE(h_values.empty());
    EXPECT_TRUE(h_col_indices.empty());
}

// Regression: FromHostData copied sizes taken from the INPUT vectors rather
// than validating them against (num_rows, nnz = values.size()). A row_offsets
// vector that was too long wrote past the num_rows+1 device allocation; one
// that was too short left the last offset as uninitialized device memory, so
// the matrix silently produced garbage rows. A col_indices/values length
// mismatch was accepted the same way.
TEST(SparseMatrixTest, FromHostDataRejectsWrongRowOffsetCount) {
    std::vector<float> values = {1.0f, 2.0f};
    std::vector<int> col_indices = {0, 1};

    std::vector<int> too_long = {0, 1, 2, 3, 4};      // needs 3 for 2 rows
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, too_long, col_indices, 2, 2),
                 std::invalid_argument);

    std::vector<int> too_short = {0, 1};              // needs 3 for 2 rows
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, too_short, col_indices, 2, 2),
                 std::invalid_argument);
}

TEST(SparseMatrixTest, FromHostDataRejectsColumnIndexCountMismatch) {
    std::vector<float> values = {1.0f, 2.0f};
    std::vector<int> row_offsets = {0, 1, 2};

    std::vector<int> too_few = {0};
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, row_offsets, too_few, 2, 2),
                 std::invalid_argument);
}

TEST(SparseMatrixTest, FromHostDataRejectsColumnIndexOutsideRange) {
    // A column index >= num_cols makes spmv read past the end of x.
    std::vector<float> values = {1.0f, 2.0f};
    std::vector<int> row_offsets = {0, 1, 2};
    std::vector<int> col_indices = {0, 5};            // 5 is out of range for 2 cols
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, row_offsets, col_indices, 2, 2),
                 std::invalid_argument);

    std::vector<int> negative = {0, -1};
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, row_offsets, negative, 2, 2),
                 std::invalid_argument);
}

TEST(SparseMatrixTest, FromHostDataRejectsInconsistentRowOffsets) {
    // row_offsets must start at 0, be non-decreasing, and end at nnz.
    std::vector<float> values = {1.0f, 2.0f};
    std::vector<int> col_indices = {0, 1};

    std::vector<int> not_from_zero = {1, 1, 2};
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, not_from_zero, col_indices, 2, 2),
                 std::invalid_argument);

    std::vector<int> decreasing = {0, 2, 1};
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, decreasing, col_indices, 2, 2),
                 std::invalid_argument);

    std::vector<int> wrong_end = {0, 1, 1};
    EXPECT_THROW(SparseMatrix<float>::FromHostData(values, wrong_end, col_indices, 2, 2),
                 std::invalid_argument);
}

TEST(SparseMatrixTest, CopyToHost) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> row_offsets = {0, 1, 2, 3, 4};
    std::vector<int> col_indices = {0, 1, 2, 3};

    auto matrix = SparseMatrix<float>::FromHostData(values, row_offsets, col_indices, 4, 4);

    std::vector<float> out_values;
    std::vector<int> out_row_offsets, out_col_indices;
    matrix.copy_to_host(out_values, out_row_offsets, out_col_indices);

    EXPECT_EQ(out_values, values);
    EXPECT_EQ(out_row_offsets, row_offsets);
    EXPECT_EQ(out_col_indices, col_indices);
}

TEST(SparseMatrixTest, DoublePrecision) {
    std::vector<double> dense = {4.0, 1.0, 1.0, 3.0};
    auto matrix = SparseMatrix<double>::FromDense(dense.data(), 2, 2);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 2);
    EXPECT_EQ(matrix->cols(), 2);
    EXPECT_EQ(matrix->nnz(), 4);
}

TEST(SparseMatrixTest, SPDMatrix) {
    std::vector<float> dense = {
        4.0f, 1.0f, 0.0f,
        1.0f, 4.0f, 1.0f,
        0.0f, 1.0f, 4.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 3);
    EXPECT_EQ(matrix->cols(), 3);
    EXPECT_EQ(matrix->nnz(), 7);
}

}  // namespace nova::sparse::test
