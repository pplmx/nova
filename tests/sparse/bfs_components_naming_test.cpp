/**
 * @file bfs_components_naming_test.cpp
 * @brief count_reachable_components counts distance-zero vertices, not components
 *
 * The function name promises connected components. The implementation is
 *
 *     for (int v = 0; v < result.num_vertices; ++v)
 *         if (result.distances[v] == 0) components++;
 *
 * and in a BFS result distance zero marks the source while -1 marks unreached. A BFSResult
 * carries distances and a visited flag, no edges, so the counting cannot change to components
 * without a different signature. The header now says what it does and marks the function
 * deprecated rather than renaming it, since nothing calls it.
 *
 * These cases pin the behaviour so a future reader can see it is one, not a component count:
 * the number returned depends only on how many distance-zero entries the caller puts in the
 * array, and is unaffected by how many vertices are unreached.
 */

#include <cuda/graph/bfs.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace bfs_components_naming_test {

// The function is deprecated, and these cases are the reason it exists. GCC's warning is
// suppressed around each call rather than at file scope so the suppression is visible next to
// what it applies to.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    TEST(BfsComponentsNamingTest, ASingleSourceCountsAsOneNoMatterHowManyAreUnreached) {
        // One source, three unreached: a graph with one component, and also one with four.
        cuda::graph::BFSResult result(4);
        result.distances[0] = 0;  // the source
        result.distances[1] = 1;
        result.distances[2] = -1;  // unreached
        result.distances[3] = -1;  // unreached

        EXPECT_EQ(cuda::graph::count_reachable_components(result), 1) << "one distance-zero vertex, regardless of how many are unreached";
    }

    TEST(BfsComponentsNamingTest, TheAnswerFollowsTheDistanceArrayNotTheGraphShape) {
        // Two entries at distance zero. A real BFS never produces this — there is one source — so
        // seeing 2 here is proof that the count reads the array rather than reasoning about
        // connectivity.
        cuda::graph::BFSResult result(3);
        result.distances[0] = 0;
        result.distances[1] = 0;
        result.distances[2] = 5;

        EXPECT_EQ(cuda::graph::count_reachable_components(result), 2) << "the function counts zero entries; it does not verify there is one source";
    }

    TEST(BfsComponentsNamingTest, AnEmptyResultCountsNothing) {
        cuda::graph::BFSResult result(0);
        EXPECT_EQ(cuda::graph::count_reachable_components(result), 0);
    }

#pragma GCC diagnostic pop

}  // namespace bfs_components_naming_test
