#include "cuda/raytrace/primitives.h"
#include "cuda/device/error.h"

namespace cuda::raytrace {

__global__ void ray_box_intersect_kernel(
    const Ray* rays,
    const AABB* boxes,
    AABB::BoxHit* results,
    size_t num_rays,
    size_t num_boxes
) {
    size_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    const Ray& ray = rays[ray_idx];
    AABB::BoxHit best_hit = {false, 0.0f, 0.0f, Vec3(0, 0, 0)};

    for (size_t i = 0; i < num_boxes; ++i) {
        auto hit = boxes[i].intersect(ray);
        if (hit.hit && (!best_hit.hit || hit.t_near < best_hit.t_near)) {
            best_hit = hit;
        }
    }

    results[ray_idx] = best_hit;
}

__global__ void ray_sphere_intersect_kernel(
    const Ray* rays,
    const Sphere* spheres,
    Sphere::SphereHit* results,
    size_t num_rays,
    size_t num_spheres
) {
    size_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    const Ray& ray = rays[ray_idx];
    Sphere::SphereHit best_hit = {false, 0.0f, 0.0f, Vec3(0, 0, 0), false};

    for (size_t i = 0; i < num_spheres; ++i) {
        auto hit = spheres[i].intersect(ray);
        if (hit.hit && (!best_hit.hit || hit.t_enter < best_hit.t_enter)) {
            best_hit = hit;
        }
    }

    results[ray_idx] = best_hit;
}

void ray_box_intersect_batch(
    const Ray* rays,
    const AABB* boxes,
    AABB::BoxHit* results,
    size_t num_rays,
    size_t num_boxes,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = static_cast<int>((num_rays + block_size - 1) / block_size);
    ray_box_intersect_kernel<<<grid_size, block_size, 0, stream>>>(
        rays, boxes, results, num_rays, num_boxes
    );
    CUDA_CHECK(cudaGetLastError());
}

void ray_sphere_intersect_batch(
    const Ray* rays,
    const Sphere* spheres,
    Sphere::SphereHit* results,
    size_t num_rays,
    size_t num_spheres,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = static_cast<int>((num_rays + block_size - 1) / block_size);
    ray_sphere_intersect_kernel<<<grid_size, block_size, 0, stream>>>(
        rays, spheres, results, num_rays, num_spheres
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda::raytrace
