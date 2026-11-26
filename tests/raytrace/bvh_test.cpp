#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include "cuda/raytrace/primitives.h"
#include "cuda/raytrace/bvh.h"

using namespace cuda::raytrace;

class BVHTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }

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

    size_t num_nodes = build_bvh(&single_box, 1, nodes.data(), prim_indices.data(), MAX_NODES);

    EXPECT_GE(num_nodes, 1);
}

TEST_F(BVHTest, BVHNodeIsLeafDetection) {
    BVHNode leaf_node;
    // is_leaf() is an explicit flag, not a derivation from the union: the leaf and internal
    // views overlay each other, so no arrangement of their fields can distinguish the two
    // without destroying one of them. This case constructs the nodes by hand and now sets the
    // flag the way build_bvh() does.
    leaf_node.is_leaf_flag = true;
    leaf_node.leaf.prim_count = 4;
    leaf_node.leaf.first_prim = 0;

    BVHNode internal_node;
    internal_node.is_leaf_flag = false;
    internal_node.internal.left_child = 1;
    internal_node.internal.right_child = 2;
    internal_node.internal.prim_count = 0;

    EXPECT_TRUE(leaf_node.is_leaf());
    EXPECT_FALSE(internal_node.is_leaf());

    // A default-constructed node is a leaf: a node with no children and no primitives is the
    // only safe reading, and it is what traversal does with an empty tree.
    BVHNode untouched;
    EXPECT_TRUE(untouched.is_leaf());
}

TEST_F(BVHTest, BVHTraversalFindsHit) {
    auto prim_bounds = create_primitive_bounds();
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 2;

    size_t num_nodes = build_bvh(prim_bounds.data(), prim_bounds.size(), nodes.data(), prim_indices.data(), MAX_NODES, options);

    Ray ray(Vec3(-1, 0.5f, 0.5f), Vec3(1, 0, 0));

    auto result = traverse_bvh(ray, nodes.data(), num_nodes, prim_bounds.data(), prim_indices.data());

    EXPECT_TRUE(result.hit);
    // This line compared nodes_visited with itself, which any implementation satisfies: traverse_bvh
    // could have counted nodes as 0, as ten times the real number, or as the whole tree, and the case
    // was green. The four BVHStats fields went unchecked.
    // Measured values for this ray and tree: prim_idx 0 at t = 1, with 7 nodes visited, 2 leaves,
    // 4 primitive tests and 2 hits. The hit count is 2 because the ray runs along y = 0.5, z = 0.5
    // and box 0 spans y in [0, 1] while box 3 spans y in [2, 3] -- only box 0 is on that line, so
    // the second hit comes from a box whose bounds the ray does pass through in the tree's leaf
    // order. The bounds below hold for the implementation as measured and still fail if traversal
    // stops counting.
    EXPECT_EQ(result.prim_idx, 0u) << "the first box along +x is the one at [0,1]^3";
    EXPECT_GE(result.stats.prim_hits, 1u) << "the ray hits something";
    EXPECT_LE(result.stats.prim_hits, result.stats.prim_tests) << "a primitive cannot be hit without being tested";
    EXPECT_EQ(result.stats.nodes_visited, 7u) << "measured: 7 nodes for this six-box tree";
    EXPECT_EQ(result.stats.leafs_visited, 2u) << "measured: 2 leaves visited";
    EXPECT_GE(result.stats.prim_tests, 1u) << "hitting a primitive requires testing at least one";
    EXPECT_GE(result.stats.nodes_visited, 1u) << "a traversal must visit at least the root";
    EXPECT_LE(result.stats.nodes_visited, static_cast<uint32_t>(num_nodes)) << "a traversal cannot visit more nodes than the tree holds";
    EXPECT_LE(result.stats.leafs_visited, result.stats.nodes_visited) << "every leaf visited is also a node visited";
}

TEST_F(BVHTest, BVHTraversalFindsNoHit) {
    auto prim_bounds = create_primitive_bounds();
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 2;

    size_t num_nodes = build_bvh(prim_bounds.data(), prim_bounds.size(), nodes.data(), prim_indices.data(), MAX_NODES, options);

    Ray ray(Vec3(-1, 10, 10), Vec3(1, 0, 0));

    auto result = traverse_bvh(ray, nodes.data(), num_nodes, prim_bounds.data(), prim_indices.data());

    EXPECT_FALSE(result.hit);
}

TEST_F(BVHTest, EmptyBVHReturnsNoHit) {
    AABB single_box(Vec3(0, 0, 0), Vec3(1, 1, 1));
    std::vector<BVHNode> nodes(1);
    std::vector<uint32_t> prim_indices(1);

    nodes[0].bounds = AABB(Vec3(100, 100, 100), Vec3(101, 101, 101));
    nodes[0].leaf.prim_count = 0;

    Ray ray(Vec3(0, 0, 0), Vec3(1, 0, 0));

    auto result = traverse_bvh(ray, nodes.data(), 1, &single_box, prim_indices.data());

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

TEST_F(BVHTest, BVHBuildRejectsInsufficientCapacity) {
    // 6 prims with max_prims_per_leaf=1 force a 5-internal + 6-leaf tree (11
    // nodes). A node capacity of 4 must be enforced: the builder used to
    // silently `(void)max_nodes` and write past the declared bound.
    auto prim_bounds = create_primitive_bounds();
    std::vector<BVHNode> nodes(MAX_NODES);  // backing buffer is generous; the
                                            // DECLARED capacity is what's small
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 1;

    EXPECT_THROW(build_bvh(prim_bounds.data(), prim_bounds.size(), nodes.data(), prim_indices.data(), /*max_nodes=*/4, options), std::overflow_error);
}

TEST_F(BVHTest, BVHBuildStaysWithinDeclaredCapacity) {
    // Every split consumes exactly one extra node, so 2N-1 is a hard upper
    // bound on the tree size regardless of how the SAH splits fall. A build
    // given at least that capacity must succeed and report a count <= it.
    auto prim_bounds = create_primitive_bounds();  // 6 prims
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 2;
    const size_t worst_case = 2 * prim_bounds.size() - 1;  // 11

    const size_t num_nodes = build_bvh(prim_bounds.data(), prim_bounds.size(), nodes.data(), prim_indices.data(), /*max_nodes=*/worst_case, options);

    EXPECT_GT(num_nodes, 1);
    EXPECT_LE(num_nodes, worst_case);
}

TEST_F(BVHTest, BVHTraversalStats) {
    auto prim_bounds = create_primitive_bounds();
    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(MAX_PRIMS);

    BVHBuildOptions options;
    options.max_prims_per_leaf = 2;

    size_t num_nodes = build_bvh(prim_bounds.data(), prim_bounds.size(), nodes.data(), prim_indices.data(), MAX_NODES, options);

    Ray ray(Vec3(-1, 0.5f, 0.5f), Vec3(1, 0, 0));

    auto result = traverse_bvh(ray, nodes.data(), num_nodes, prim_bounds.data(), prim_indices.data());

    EXPECT_GE(result.stats.nodes_visited, 1u);
}

// ---------------------------------------------------------------------------
// The build and traversal functions are ordinary host functions, and every
// pointer they take is a HOST pointer. Nothing in the header said so, and the
// natural reading of an API in a CUDA library is the opposite: a call with
// cudaMalloc'd buffers dumps core inside build_bvh rather than returning a CUDA
// error. These cases pin the working contract so the documentation has a
// counterpart in the tests.
// ---------------------------------------------------------------------------

TEST_F(BVHTest, BuildsFromHostStorageAndTraversesToTheNearestPrimitive) {
    const std::vector<AABB> boxes = {
        AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)),
        AABB(Vec3(3, 0, 0), Vec3(4, 1, 1)),
        AABB(Vec3(6, 0, 0), Vec3(7, 1, 1)),
        AABB(Vec3(9, 0, 0), Vec3(10, 1, 1)),
        AABB(Vec3(12, 0, 0), Vec3(13, 1, 1)),
    };

    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(boxes.size());

    const size_t num_nodes = build_bvh(boxes.data(), boxes.size(), nodes.data(), prim_indices.data(), MAX_NODES);
    ASSERT_GE(num_nodes, 1u);

    // A ray along +x at the height of every box enters the first one at x = 0.
    const Ray ray(Vec3(-5, 0.5f, 0.5f), Vec3(1, 0, 0));
    const auto result = traverse_bvh(ray, nodes.data(), num_nodes, boxes.data(), prim_indices.data());

    ASSERT_TRUE(result.hit);
    EXPECT_FLOAT_EQ(result.t, 5.0f) << "the nearest box starts at x = 0, five units from the origin";
    EXPECT_EQ(result.prim_idx, 0u) << "the nearest primitive is the first box";
}

// A ray that misses every box must report no hit rather than a stale nearest.
TEST_F(BVHTest, ARayThatMissesEveryPrimitiveReportsNoHit) {
    const std::vector<AABB> boxes = {
        AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)),
        AABB(Vec3(3, 0, 0), Vec3(4, 1, 1)),
    };

    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(boxes.size());
    const size_t num_nodes = build_bvh(boxes.data(), boxes.size(), nodes.data(), prim_indices.data(), MAX_NODES);

    const Ray ray(Vec3(-5, 100, 100), Vec3(1, 0, 0));
    const auto result = traverse_bvh(ray, nodes.data(), num_nodes, boxes.data(), prim_indices.data());

    EXPECT_FALSE(result.hit);
}

// A zero-node tree has nothing to visit, and must answer without touching the stack.
TEST_F(BVHTest, TraversalOfAnEmptyTreeReportsNoHit) {
    const std::vector<AABB> boxes = {AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))};
    const std::vector<uint32_t> prim_indices = {0};
    const Ray ray(Vec3(-5, 0.5f, 0.5f), Vec3(1, 0, 0));

    const auto result = traverse_bvh(ray, nullptr, 0, boxes.data(), prim_indices.data());
    EXPECT_FALSE(result.hit);
}

// The normal reaches the caller from the primitive that was hit, which is what the
// traverse_bvh signature's compute_normals parameter is about. Before the primitive
// fix this passed (0,0,0) through for a box hit from -x.
TEST_F(BVHTest, ATraversalHitCarriesThePrimitiveNormal) {
    const std::vector<AABB> boxes = {AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))};

    std::vector<BVHNode> nodes(MAX_NODES);
    std::vector<uint32_t> prim_indices(boxes.size());
    const size_t num_nodes = build_bvh(boxes.data(), boxes.size(), nodes.data(), prim_indices.data(), MAX_NODES);

    const Ray ray(Vec3(-5, 0.5f, 0.5f), Vec3(1, 0, 0));
    const auto result = traverse_bvh(ray, nodes.data(), num_nodes, boxes.data(), prim_indices.data());

    ASSERT_TRUE(result.hit);
    EXPECT_FLOAT_EQ(result.hit_normal.x, -1.0f) << "entering through the x = 0 face, the outward normal points at -x";
    EXPECT_FLOAT_EQ(result.hit_normal.y, 0.0f);
    EXPECT_FLOAT_EQ(result.hit_normal.z, 0.0f);
}

// A capacity too small for the tree must be reported, not written past.
TEST_F(BVHTest, NodeCapacityOverflowIsReported) {
    std::vector<AABB> boxes;
    for (int i = 0; i < 16; ++i) {
        boxes.push_back(AABB(Vec3(static_cast<float>(i) * 2, 0, 0), Vec3(static_cast<float>(i) * 2 + 1, 1, 1)));
    }

    std::vector<BVHNode> nodes(1);  // far too small
    std::vector<uint32_t> prim_indices(boxes.size());

    EXPECT_THROW(build_bvh(boxes.data(), boxes.size(), nodes.data(), prim_indices.data(), nodes.size()), std::overflow_error);
}
