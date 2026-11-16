/**
 * @file pagerank_result_test.cpp
 * @brief Contracts for the PageRank result container's lifetime states
 *
 * include/cuda/graph/pagerank.h owns raw host and device pointers and is
 * move-only. top_vertex() read ranks[0] without checking that any ranks exist, so
 * two ordinary lifetime states crashed or read out of bounds:
 *
 *   PageRankResult moved_from(4);
 *   PageRankResult owner(std::move(moved_from));
 *   moved_from.top_vertex();   // ranks is nullptr -> measured "dumped core"
 *
 *   PageRankResult empty(0);
 *   empty.top_vertex();        // ranks is a zero-length allocation, reads past it
 *
 * A moved-from object is a legal state to destroy and to assign into, and a
 * zero-vertex result is a legitimate answer for an empty graph, so neither should
 * be fatal. top_vertex() now returns -1 when there is no vertex to report, and the
 * header documents that.
 */

#include <cuda/graph/pagerank.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <utility>
#include <vector>

namespace cuda::graph::test {

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

// Regression: this dereferenced a null ranks pointer.
TEST(PageRankResultTest, MovedFromObjectCanBeQueriedSafely) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    PageRankResult original(4);
    PageRankResult owner(std::move(original));

    // The moved-from object is in a valid state for destruction and assignment.
    EXPECT_EQ(original.num_vertices, 0);
    EXPECT_EQ(original.ranks, nullptr);

    EXPECT_EQ(original.top_vertex(), -1)
        << "a moved-from result has no vertices to report";
}

// Regression: this read past a zero-length allocation.
TEST(PageRankResultTest, ZeroVerticesReportNoTopVertex) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const PageRankResult empty(0);

    EXPECT_EQ(empty.num_vertices, 0);
    EXPECT_EQ(empty.top_vertex(), -1)
        << "an empty graph has no highest-ranked vertex";
}

TEST(PageRankResultTest, ClearedResultCanBeQueriedSafely) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    PageRankResult result(8);
    result.clear();

    EXPECT_EQ(result.num_vertices, 0);
    EXPECT_EQ(result.ranks, nullptr);
    EXPECT_EQ(result.top_vertex(), -1);
}

TEST(PageRankResultTest, TopVertexFindsTheMaximumRank) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    PageRankResult result(5);
    // Distinct values so the winner is unambiguous.
    const std::vector<float> ranks = {0.1f, 0.05f, 0.6f, 0.15f, 0.1f};
    for (int i = 0; i < 5; ++i) {
        result.ranks[i] = ranks[static_cast<size_t>(i)];
    }

    EXPECT_EQ(result.top_vertex(), 2) << "index 2 holds the largest rank";
}

TEST(PageRankResultTest, TopVertexPicksTheFirstWhenRanksTie) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    PageRankResult result(3);
    // A strict greater-than comparison keeps the earliest index on a tie, which
    // makes the result deterministic for symmetric graphs.
    result.ranks[0] = 0.5f;
    result.ranks[1] = 0.5f;
    result.ranks[2] = 0.5f;

    EXPECT_EQ(result.top_vertex(), 0);
}

TEST(PageRankResultTest, MoveTransfersOwnershipExactlyOnce) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    PageRankResult first(4);
    first.ranks[0] = 1.0f;
    first.ranks[1] = 2.0f;
    first.ranks[2] = 3.0f;
    first.ranks[3] = 4.0f;
    first.iterations = 7;
    first.final_delta = 0.25f;

    PageRankResult second(std::move(first));

    EXPECT_EQ(second.num_vertices, 4);
    EXPECT_EQ(second.top_vertex(), 3) << "index 3 holds the largest rank";
    EXPECT_EQ(second.iterations, 7);
    EXPECT_NEAR(second.final_delta, 0.25f, 1e-6f);

    // The source must not still point at the transferred buffer.
    EXPECT_EQ(first.ranks, nullptr);
    EXPECT_EQ(first.num_vertices, 0);
}

TEST(PageRankResultTest, MoveAssignmentReleasesThePreviousBuffer) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    PageRankResult source(2);
    source.ranks[0] = 0.0f;
    source.ranks[1] = 9.0f;

    PageRankResult target(6);
    target = std::move(source);

    EXPECT_EQ(target.num_vertices, 2)
        << "the assignment must replace the old size, not merge";
    EXPECT_EQ(target.top_vertex(), 1);
    EXPECT_EQ(source.ranks, nullptr);
}

TEST(PageRankResultTest, UploadAndDownloadRoundTrip) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    PageRankResult result(4);
    const std::vector<float> ranks = {0.1f, 0.4f, 0.2f, 0.3f};
    for (int i = 0; i < 4; ++i) {
        result.ranks[i] = ranks[static_cast<size_t>(i)];
    }

    ASSERT_NO_THROW(result.upload());
    // Scramble the host copy, then read it back from the device.
    for (int i = 0; i < 4; ++i) {
        result.ranks[i] = -1.0f;
    }
    ASSERT_NO_THROW(result.download());

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.ranks[i], ranks[static_cast<size_t>(i)], 1e-6f)
            << "element " << i;
    }
}

TEST(PageRankResultTest, DefaultConstructedIsEmpty) {
    const PageRankResult result;

    EXPECT_EQ(result.num_vertices, 0);
    EXPECT_EQ(result.ranks, nullptr);
    EXPECT_EQ(result.top_vertex(), -1);
}

}  // namespace cuda::graph::test
