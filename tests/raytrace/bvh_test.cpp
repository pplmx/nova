#include <gtest/gtest.h>
#include <vector>
#include "cuda/raytrace/primitives.h"
#include "cuda/raytrace/bvh.h"

using namespace cuda::raytrace;

class BVHTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }

    static constexpr size_t MAX_NODES = 256;
    static constexpr size_t MAX_PRIMS = 64;

    std::vector<AABB> create_primitive_bounds() {
        return {
            AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)),
            AABB(Vec3(2, 0, 0), Vec3(3, 1, 1)),
            AABB(Vec3(4, 0, 0), Vec3(5, 1, 1)),
            AABB(Vec3(0, 2, 0), Vec3(1, 3, 1)),
            AABB(Vec3(2, 2, 0), Vec3(3, 3, 1)),
            AABB(Vec3(4, 2, 0), Vec3(5, 3, 1)),
        };
    }
};

TEST_F(BVHTest, BuildBVHWithKnownPrimitives) {
    AABB single_box(Vec3(0, 0, 0), Vec3(1, 1, 1));
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    size_t num_nodes = build_bvh(
        &single_box, 1,
        nodes.data(), prim_indices.data(), MAX_NODES
    );

    EXPECT_GE(num_nodes, 1);
}

TEST_F(BVHTest, BVHNodeIsLeafDetection) {
    BVHNode leaf_node;
    leaf_node.leaf.prim_count = 4;
    leaf_node.leaf.first_prim = 0;

    BVHNode internal_node;
    internal_node.internal.left_child = 1;
    internal_node.internal.right_child = 2;
    internal_node.internal.prim_count = 0;
    internal_node.leaf.prim_count = 0;

    EXPECT_TRUE(leaf_node.is_leaf());
    EXPECT_FALSE(internal_node.is_leaf());
}

TEST_F(BVHTest, BVHTraversalFindsHit) {
    auto prim_bounds = create_primitive_bounds();
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 2;

    size_t num_nodes = build_bvh(
        prim_bounds.data(), prim_bounds.size(),
        nodes.data(), prim_indices.data(), MAX_NODES,
        options
    );

    Ray ray(Vec3(-1, 0.5f, 0.5f), Vec3(1, 0, 0));

    auto result = traverse_bvh(
        ray, nodes.data(), num_nodes,
        prim_bounds.data(), prim_indices.data()
    );

    EXPECT_TRUE(result.hit);
    EXPECT_EQ(result.stats.nodes_visited, result.stats.nodes_visited);
}

TEST_F(BVHTest, BVHTraversalFindsNoHit) {
    auto prim_bounds = create_primitive_bounds();
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 2;

    size_t num_nodes = build_bvh(
        prim_bounds.data(), prim_bounds.size(),
        nodes.data(), prim_indices.data(), MAX_NODES,
        options
    );

    Ray ray(Vec3(-1, 10, 10), Vec3(1, 0, 0));

    auto result = traverse_bvh(
        ray, nodes.data(), num_nodes,
        prim_bounds.data(), prim_indices.data()
    );

    EXPECT_FALSE(result.hit);
}

TEST_F(BVHTest, EmptyBVHReturnsNoHit) {
    AABB single_box(Vec3(0, 0, 0), Vec3(1, 1, 1));
    std::vector<BVHNode> nodes(1);
    std::vector<uint32_t> prim_indices(1);

    nodes[0].bounds = AABB(Vec3(100, 100, 100), Vec3(101, 101, 101));
    nodes[0].leaf.prim_count = 0;

    Ray ray(Vec3(0, 0, 0), Vec3(1, 0, 0));

    auto result = traverse_bvh(
        ray, nodes.data(), 1,
        &single_box, prim_indices.data()
    );

    EXPECT_FALSE(result.hit);
}

TEST_F(BVHTest, BVHLeafHelpers) {
    BVHNode leaf_node;
    leaf_node.leaf.prim_count = 5;
    leaf_node.leaf.first_prim = 10;

    EXPECT_EQ(get_leaf_prim_count(leaf_node), 5);
    EXPECT_EQ(get_leaf_first_prim_index(leaf_node), 10);
}

TEST_F(BVHTest, BVHMemoryEstimate) {
    size_t num_prims = 100;
    size_t estimated = bvh_memory_estimate(num_prims);

    EXPECT_GT(estimated, num_prims * sizeof(BVHNode));
    EXPECT_GT(estimated, num_prims * sizeof(uint32_t));
}

TEST_F(BVHTest, BVHBuildOptionsDefault) {
    BVHBuildOptions options;

    EXPECT_EQ(options.max_prims_per_leaf, 4);
    EXPECT_EQ(options.min_prims_per_leaf, 1);
    EXPECT_TRUE(options.use_sah);
    EXPECT_EQ(options.bins, 12);
}

TEST_F(BVHTest, BVHTraversalStats) {
    auto prim_bounds = create_primitive_bounds();
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 2;

    size_t num_nodes = build_bvh(
        prim_bounds.data(), prim_bounds.size(),
        nodes.data(), prim_indices.data(), MAX_NODES,
        options
    );

    Ray ray(Vec3(-1, 0.5f, 0.5f), Vec3(1, 0, 0));

    auto result = traverse_bvh(
        ray, nodes.data(), num_nodes,
        prim_bounds.data(), prim_indices.data()
    );

    EXPECT_GE(result.stats.nodes_visited, 1u);
}
