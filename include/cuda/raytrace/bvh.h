#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "cuda/raytrace/primitives.h"

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

    __host__ __device__ bool is_leaf() const {
        return internal.prim_count > 0 || leaf.prim_count > 0;
    }
};

struct BVHBuildOptions {
    int max_prims_per_leaf = 4;
    int min_prims_per_leaf = 1;
    bool use_sah = true;
    int bins = 12;
};

size_t build_bvh(
    const AABB* prim_bounds,
    size_t num_prims,
    BVHNode* nodes,
    uint32_t* prim_indices,
    size_t max_nodes,
    const BVHBuildOptions& options = BVHBuildOptions{},
    cudaStream_t stream = nullptr
);

size_t build_bvh_spheres(
    const Sphere* spheres,
    size_t num_spheres,
    BVHNode* nodes,
    uint32_t* prim_indices,
    size_t max_nodes,
    const BVHBuildOptions& options = BVHBuildOptions{},
    cudaStream_t stream = nullptr
);

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

BVHTraversalResult traverse_bvh(
    const Ray& ray,
    const BVHNode* nodes,
    size_t num_nodes,
    const AABB* prim_bounds,
    const uint32_t* prim_indices,
    bool compute_normals = true,
    cudaStream_t stream = nullptr
);

BVHTraversalResult traverse_bvh_spheres(
    const Ray& ray,
    const BVHNode* nodes,
    size_t num_nodes,
    const Sphere* spheres,
    const uint32_t* prim_indices,
    cudaStream_t stream = nullptr
);

int get_leaf_prim_count(const BVHNode& node);

uint32_t get_leaf_first_prim_index(const BVHNode& node);

bool ray_hits_bounds(const Ray& ray, const AABB& bounds);

size_t bvh_memory_estimate(size_t num_prims);

}  // namespace cuda::raytrace
