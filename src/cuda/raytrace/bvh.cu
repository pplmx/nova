#include "cuda/raytrace/bvh.h"
#include "cuda/raytrace/primitives.h"
#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>

namespace cuda::raytrace {

constexpr int SAH_BIN_COUNT = 12;

struct SAHBin {
    AABB bounds;
    int prim_count;
};

struct SAHPartition {
    int dim;
    float pos;
    AABB left_bounds;
    AABB right_bounds;
    int left_count;
    int right_count;
    float cost;
};

int get_leaf_prim_count(const BVHNode& node) {
    return node.is_leaf() ? node.leaf.prim_count : 0;
}

uint32_t get_leaf_first_prim_index(const BVHNode& node) {
    return node.is_leaf() ? node.leaf.first_prim : 0;
}

__host__ __device__ bool ray_hits_bounds(const Ray& ray, const AABB& bounds) {
    float tmin = ray.t_min;
    float tmax = ray.t_max;

    float inv_dir_x = (ray.direction.x != 0.0f) ? 1.0f / ray.direction.x : 1e30f;
    float inv_dir_y = (ray.direction.y != 0.0f) ? 1.0f / ray.direction.y : 1e30f;
    float inv_dir_z = (ray.direction.z != 0.0f) ? 1.0f / ray.direction.z : 1e30f;

    float t1, t2;

    t1 = (bounds.min_point.x - ray.origin.x) * inv_dir_x;
    t2 = (bounds.max_point.x - ray.origin.x) * inv_dir_x;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));
    if (tmin > tmax) return false;

    t1 = (bounds.min_point.y - ray.origin.y) * inv_dir_y;
    t2 = (bounds.max_point.y - ray.origin.y) * inv_dir_y;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));
    if (tmin > tmax) return false;

    t1 = (bounds.min_point.z - ray.origin.z) * inv_dir_z;
    t2 = (bounds.max_point.z - ray.origin.z) * inv_dir_z;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));

    return tmin <= tmax;
}

namespace {

SAHPartition find_best_sah_partition(
    const AABB* prim_bounds,
    const uint32_t* prim_indices,
    int start,
    int end,
    const BVHBuildOptions& options
) {
    SAHPartition best = {-1, 0, AABB(), AABB(), 0, 0, 1e30f};

    AABB total_bounds = prim_bounds[prim_indices[start]];
    for (int i = start + 1; i < end; ++i) {
        total_bounds.expand(prim_bounds[prim_indices[i]]);
    }

    float axis_min, axis_max;
    float range;

    for (int dim = 0; dim < 3; ++dim) {
        if (dim == 0) {
            axis_min = total_bounds.min_point.x;
            axis_max = total_bounds.max_point.x;
        } else if (dim == 1) {
            axis_min = total_bounds.min_point.y;
            axis_max = total_bounds.max_point.y;
        } else {
            axis_min = total_bounds.min_point.z;
            axis_max = total_bounds.max_point.z;
        }

        range = axis_max - axis_min;
        if (range < 1e-6f) continue;

        SAHBin bins[SAH_BIN_COUNT];
        for (int i = 0; i < SAH_BIN_COUNT; ++i) {
            bins[i].prim_count = 0;
        }

        for (int i = start; i < end; ++i) {
            float centroid;
            const AABB& box = prim_bounds[prim_indices[i]];
            if (dim == 0) {
                centroid = (box.min_point.x + box.max_point.x) * 0.5f;
            } else if (dim == 1) {
                centroid = (box.min_point.y + box.max_point.y) * 0.5f;
            } else {
                centroid = (box.min_point.z + box.max_point.z) * 0.5f;
            }

            int bin = static_cast<int>((centroid - axis_min) / range * (SAH_BIN_COUNT - 1));
            bin = max(0, min(SAH_BIN_COUNT - 1, bin));

            if (bins[bin].prim_count == 0) {
                bins[bin].bounds = box;
            } else {
                bins[bin].bounds.expand(box);
            }
            bins[bin].prim_count++;
        }

        AABB left_bounds;
        AABB right_bounds;
        int left_count = 0;
        int right_count = end - start;

        for (int i = 0; i < SAH_BIN_COUNT - 1; ++i) {
            left_count += bins[i].prim_count;
            right_count -= bins[i].prim_count;

            if (left_count == 0 || right_count == 0) continue;

            if (i == 0 || bins[i].prim_count == 1) {
                left_bounds = bins[i].bounds;
            } else {
                left_bounds.expand(bins[i].bounds);
            }

            Vec3 left_extent = left_bounds.extent();
            Vec3 right_extent = right_bounds.extent();
            float left_sa = 2.0f * (left_extent.x * left_extent.y +
                                    left_extent.y * left_extent.z +
                                    left_extent.z * left_extent.x);
            float right_sa = 2.0f * (right_extent.x * right_extent.y +
                                      right_extent.y * right_extent.z +
                                      right_extent.z * right_extent.x);

            float cost = left_count * left_sa + right_count * right_sa;

            if (cost < best.cost) {
                best.dim = dim;
                best.left_count = left_count;
                best.right_count = right_count;
                best.cost = cost;
                best.left_bounds = left_bounds;
                best.right_bounds = right_bounds;
                best.pos = axis_min + (range * (i + 1) / SAH_BIN_COUNT);
            }
        }
    }

    return best;
}

size_t build_bvh_recursive(
    const AABB* prim_bounds,
    uint32_t* prim_indices,
    BVHNode* nodes,
    size_t node_index,
    int start,
    int end,
    const BVHBuildOptions& options,
    size_t& next_node
) {
    BVHNode& node = nodes[node_index];
    int prim_count = end - start;

    node.bounds = prim_bounds[prim_indices[start]];
    for (int i = start + 1; i < end; ++i) {
        node.bounds.expand(prim_bounds[prim_indices[i]]);
    }

    if (prim_count <= options.max_prims_per_leaf) {
        node.leaf.prim_count = prim_count;
        node.leaf.first_prim = start;
        return next_node;
    }

    SAHPartition partition;
    if (options.use_sah) {
        partition = find_best_sah_partition(prim_bounds, prim_indices, start, end, options);
    }

    int mid;
    if (partition.dim < 0) {
        mid = start + prim_count / 2;
    } else {
        mid = start + partition.left_count;
        float split_pos = partition.pos;
        int left_idx = start;
        int right_idx = mid;

        for (int i = start; i < end; ++i) {
            float centroid;
            const AABB& box = prim_bounds[prim_indices[i]];
            if (partition.dim == 0) {
                centroid = (box.min_point.x + box.max_point.x) * 0.5f;
            } else if (partition.dim == 1) {
                centroid = (box.min_point.y + box.max_point.y) * 0.5f;
            } else {
                centroid = (box.min_point.z + box.max_point.z) * 0.5f;
            }

            if (centroid > split_pos && right_idx < end) {
                uint32_t temp = prim_indices[i];
                prim_indices[i] = prim_indices[right_idx];
                prim_indices[right_idx] = temp;
                right_idx++;
            }
        }

        if (left_idx == start || right_idx == end) {
            mid = start + prim_count / 2;
        }
    }

    node.internal.left_child = next_node++;
    node.internal.right_child = next_node++;
    node.internal.prim_count = 0;

    build_bvh_recursive(prim_bounds, prim_indices, nodes, node.internal.left_child,
                        start, mid, options, next_node);
    build_bvh_recursive(prim_bounds, prim_indices, nodes, node.internal.right_child,
                        mid, end, options, next_node);

    return next_node;
}

}  // anonymous namespace

size_t build_bvh(
    const AABB* prim_bounds,
    size_t num_prims,
    BVHNode* nodes,
    uint32_t* prim_indices,
    size_t max_nodes,
    const BVHBuildOptions& options,
    cudaStream_t
) {
    (void)max_nodes;

    for (size_t i = 0; i < num_prims; ++i) {
        prim_indices[i] = i;
    }

    size_t next_node = 1;
    size_t num_nodes = build_bvh_recursive(
        prim_bounds, prim_indices, nodes, 0, 0, static_cast<int>(num_prims), options, next_node
    );

    return num_nodes;
}

size_t build_bvh_spheres(
    const Sphere* spheres,
    size_t num_spheres,
    BVHNode* nodes,
    uint32_t* prim_indices,
    size_t max_nodes,
    const BVHBuildOptions& options,
    cudaStream_t stream
) {
    (void)stream;

    AABB* bounds = new AABB[num_spheres];
    for (size_t i = 0; i < num_spheres; ++i) {
        Vec3 min_p(spheres[i].center.x - spheres[i].radius,
                   spheres[i].center.y - spheres[i].radius,
                   spheres[i].center.z - spheres[i].radius);
        Vec3 max_p(spheres[i].center.x + spheres[i].radius,
                   spheres[i].center.y + spheres[i].radius,
                   spheres[i].center.z + spheres[i].radius);
        bounds[i] = AABB(min_p, max_p);
    }

    BVHBuildOptions local_options = options;
    size_t next_node = 1;
    for (size_t i = 0; i < num_spheres; ++i) {
        prim_indices[i] = i;
    }
    size_t num_nodes = build_bvh_recursive(
        bounds, prim_indices, nodes, 0, 0, static_cast<int>(num_spheres), local_options, next_node
    );

    delete[] bounds;
    return num_nodes;
}

BVHTraversalResult traverse_bvh(
    const Ray& ray,
    const BVHNode* nodes,
    size_t num_nodes,
    const AABB* prim_bounds,
    const uint32_t* prim_indices,
    bool,
    cudaStream_t
) {
    BVHTraversalResult result = {false, 0.0f, Vec3(), Vec3(), 0, BVHStats()};

    if (num_nodes == 0) return result;

    constexpr int MAX_STACK = 64;
    int stack[MAX_STACK];
    int stack_top = 0;
    stack[stack_top++] = 0;

    while (stack_top > 0) {
        stack_top--;
        int node_idx = stack[stack_top];

        if (node_idx >= static_cast<int>(num_nodes)) continue;

        const BVHNode& node = nodes[node_idx];
        result.stats.nodes_visited++;

        if (!ray_hits_bounds(ray, node.bounds)) {
            continue;
        }

        if (node.is_leaf()) {
            result.stats.leafs_visited++;
            int prim_count = node.leaf.prim_count;
            int first_prim = node.leaf.first_prim;

            for (int i = 0; i < prim_count; ++i) {
                int prim_idx = prim_indices[first_prim + i];
                result.stats.prim_tests++;

                auto hit = prim_bounds[prim_idx].intersect(ray);
                if (hit.hit && (!result.hit || hit.t_near < result.t)) {
                    result.hit = true;
                    result.t = hit.t_near;
                    result.hit_point = ray.point_at(hit.t_near);
                    result.hit_normal = hit.hit_normal;
                    result.prim_idx = prim_idx;
                    result.stats.prim_hits++;
                }
            }
        } else {
            int left_idx = node.internal.left_child;
            int right_idx = node.internal.right_child;

            if (left_idx < static_cast<int>(num_nodes) && right_idx < static_cast<int>(num_nodes)) {
                if (stack_top < MAX_STACK - 1) stack[stack_top++] = left_idx;
                if (stack_top < MAX_STACK - 1) stack[stack_top++] = right_idx;
            }
        }
    }

    return result;
}

BVHTraversalResult traverse_bvh_spheres(
    const Ray& ray,
    const BVHNode* nodes,
    size_t num_nodes,
    const Sphere* spheres,
    const uint32_t* prim_indices,
    cudaStream_t stream
) {
    BVHTraversalResult result = {false, 0.0f, Vec3(), Vec3(), 0, BVHStats()};

    if (num_nodes == 0) return result;

    constexpr int MAX_STACK = 64;
    int stack[MAX_STACK];
    int stack_top = 0;
    stack[stack_top++] = 0;

    while (stack_top > 0) {
        stack_top--;
        int node_idx = stack[stack_top];

        if (node_idx >= static_cast<int>(num_nodes)) continue;

        const BVHNode& node = nodes[node_idx];
        result.stats.nodes_visited++;

        if (!ray_hits_bounds(ray, node.bounds)) {
            continue;
        }

        if (node.is_leaf()) {
            result.stats.leafs_visited++;
            int prim_count = node.leaf.prim_count;
            int first_prim = node.leaf.first_prim;

            for (int i = 0; i < prim_count; ++i) {
                int prim_idx = prim_indices[first_prim + i];
                result.stats.prim_tests++;

                auto hit = spheres[prim_idx].intersect(ray);
                if (hit.hit && (!result.hit || hit.t_enter < result.t)) {
                    result.hit = true;
                    result.t = hit.t_enter;
                    result.hit_point = ray.point_at(hit.t_enter);
                    result.hit_normal = hit.hit_normal;
                    result.prim_idx = prim_idx;
                    result.stats.prim_hits++;
                }
            }
        } else {
            int left_idx = node.internal.left_child;
            int right_idx = node.internal.right_child;

            if (left_idx < static_cast<int>(num_nodes) && right_idx < static_cast<int>(num_nodes)) {
                if (stack_top < MAX_STACK - 1) stack[stack_top++] = left_idx;
                if (stack_top < MAX_STACK - 1) stack[stack_top++] = right_idx;
            }
        }
    }

    return result;
}

size_t bvh_memory_estimate(size_t num_prims) {
    size_t num_nodes = num_prims * 2;
    return num_nodes * sizeof(BVHNode) + num_prims * sizeof(uint32_t);
}

}  // namespace cuda::raytrace
