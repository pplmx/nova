#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "cuda/raytrace/primitives.h"

/**
 * @file bvh.h
 * @brief Bounding volume hierarchy build and traversal
 *
 * @par Every pointer in this header is a HOST pointer
 * build_bvh(), build_bvh_spheres() and traverse_bvh() are ordinary host functions -- none of
 * them is __global__ -- and they dereference their arguments directly on the CPU. A caller
 * that passes cudaMalloc'd buffers gets a host SIGSEGV at the first element, not a CUDA
 * error. Measured: a five-box build with cudaMalloc'd prim_bounds dumped core inside
 * build_bvh, while the same call with std::vector storage returned a tree.
 *
 * The CUDA-stream parameter on each of them is accepted and ignored; this is a host-side
 * builder, and the tests are the only current caller.
 */

namespace cuda::raytrace {

    struct BVHNode {
        AABB bounds;

        union {
            struct {
                uint32_t left_child;
                uint32_t right_child;
                uint32_t prim_count;
                uint32_t first_prim;
            } internal;

            struct {
                uint32_t prim_count;
                uint32_t first_prim;
                uint32_t pad1;
                uint32_t pad2;
            } leaf;
        };

        /// Whether this node has no children.
        ///
        /// This cannot be derived from the union. Two separately broken definitions lived
        /// here, and the second only surfaced after fixing the first:
        ///
        ///     return internal.prim_count > 0 || leaf.prim_count > 0;   // original
        ///     return internal.left_child == 0;                        // first attempt
        ///
        /// The original was wrong because `internal.prim_count` is the THIRD word and is 0 on
        /// both kinds of node, so the OR fell through to `leaf.prim_count`, which overlays
        /// `internal.left_child` (the FIRST word) and is therefore a child index on any
        /// internal node -- always >= 1. Every internal node claimed to be a leaf and the tree
        /// was never descended: measured on five boxes along x, a ray that should have hit box
        /// 0 at t = 5 reported box 2 at t = 11 after a single primitive test.
        ///
        /// The first attempt moved the test to `internal.left_child == 0`, which is true for a
        /// leaf only if the builder zeroes that word. It cannot: the same word has to hold
        /// `leaf.prim_count`, so zeroing one destroys the other, and the tree either loses its
        /// leaf counts or grows a cycle through child index 0.
        ///
        /// The flag is therefore explicit and lives outside the union.
        bool is_leaf_flag = true;

        __host__ __device__ bool is_leaf() const { return is_leaf_flag; }
    };

    struct BVHBuildOptions {
        int max_prims_per_leaf = 4;
        int min_prims_per_leaf = 1;
        bool use_sah = true;
        int bins = 12;
    };

    /**
     * @brief Build a BVH over the given primitive bounds
     *
     * @param prim_bounds        Host array of num_prims boxes
     * @param num_prims          Number of primitives
     * @param nodes              Host array of max_nodes nodes, written by this call
     * @param prim_indices       Host array of num_prims indices, reordered by this call
     * @param max_nodes          Capacity of @p nodes; the tree throws std::overflow_error
     *                           rather than writing past it
     * @param options            Build options
     * @param stream             Ignored; this is a host-side builder
     * @return Number of nodes the tree occupies
     */
    size_t build_bvh(const AABB* prim_bounds,
                     size_t num_prims,
                     BVHNode* nodes,
                     uint32_t* prim_indices,
                     size_t max_nodes,
                     const BVHBuildOptions& options = BVHBuildOptions {},
                     cudaStream_t stream = nullptr);

    /**
     * @brief Build a BVH over spheres, using each sphere's enclosing box
     *
     * Same pointer and stream contract as build_bvh().
     */
    size_t build_bvh_spheres(const Sphere* spheres,
                             size_t num_spheres,
                             BVHNode* nodes,
                             uint32_t* prim_indices,
                             size_t max_nodes,
                             const BVHBuildOptions& options = BVHBuildOptions {},
                             cudaStream_t stream = nullptr);

    struct BVHStats {
        uint32_t nodes_visited = 0;
        uint32_t leafs_visited = 0;
        uint32_t prim_tests = 0;
        uint32_t prim_hits = 0;
    };

    struct BVHTraversalResult {
        bool hit;
        float t;
        Vec3 hit_point;
        Vec3 hit_normal;
        uint32_t prim_idx;
        BVHStats stats;
    };

    /**
     * @brief Find the nearest hit for one ray
     *
     * @param ray            Ray to trace
     * @param nodes          Host array produced by build_bvh()
     * @param num_nodes      Node count returned by build_bvh()
     * @param prim_bounds    Host array of the same boxes passed to build_bvh()
     * @param prim_indices   Host index array produced by build_bvh()
     * @param compute_normals Ignored; the normal always comes from the primitive's own
     *                        intersect()
     * @param stream         Ignored; this is a host-side traversal
     */
    BVHTraversalResult traverse_bvh(
        const Ray& ray, const BVHNode* nodes, size_t num_nodes, const AABB* prim_bounds, const uint32_t* prim_indices, bool compute_normals = true, cudaStream_t stream = nullptr);

    /**
     * @brief Find the nearest hit for one ray against a sphere BVH
     *
     * Host pointers, same contract as traverse_bvh().
     */
    BVHTraversalResult traverse_bvh_spheres(
        const Ray& ray, const BVHNode* nodes, size_t num_nodes, const Sphere* spheres, const uint32_t* prim_indices, cudaStream_t stream = nullptr);

    int get_leaf_prim_count(const BVHNode& node);

    uint32_t get_leaf_first_prim_index(const BVHNode& node);

    bool ray_hits_bounds(const Ray& ray, const AABB& bounds);

    size_t bvh_memory_estimate(size_t num_prims);

}  // namespace cuda::raytrace
