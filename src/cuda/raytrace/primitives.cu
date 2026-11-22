#include "cuda/raytrace/primitives.h"
#include "cuda/device/error.h"

namespace cuda::raytrace {

    __global__ void ray_box_intersect_kernel(const Ray* rays, const AABB* boxes, AABB::BoxHit* results, size_t num_rays, size_t num_boxes) {
        size_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= num_rays) {
            return;
        }

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

    __global__ void ray_sphere_intersect_kernel(const Ray* rays, const Sphere* spheres, Sphere::SphereHit* results, size_t num_rays, size_t num_spheres) {
        size_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= num_rays) {
            return;
        }

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

    namespace {

        /// Write the "hit nothing" answer to every element of a results array.
        ///
        /// The zero-primitive case below cannot launch a kernel (a zero-sized grid is
        /// "invalid configuration argument"), but returning without writing anything left the
        /// caller's buffer holding whatever it held before -- measured, a poisoned entry read
        /// back as hit == true with the poison's t_near. The header promises one BoxHit per ray
        /// and that "a ray that hits nothing gets hit == false", which for zero boxes is every
        /// ray.
        __global__ void fill_missed_box_hits_kernel(AABB::BoxHit* results, size_t num_rays) {
            const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < num_rays) {
                results[i] = AABB::BoxHit {false, 0.0f, 0.0f, Vec3(0, 0, 0)};
            }
        }

        __global__ void fill_missed_sphere_hits_kernel(Sphere::SphereHit* results, size_t num_rays) {
            const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < num_rays) {
                results[i] = Sphere::SphereHit {false, 0.0f, 0.0f, Vec3(0, 0, 0), false};
            }
        }

        constexpr int kRayBlockSize = 256;

    }  // namespace

    void ray_box_intersect_batch(const Ray* rays, const AABB* boxes, AABB::BoxHit* results, size_t num_rays, size_t num_boxes, cudaStream_t stream) {
        // A zero-ray batch produces a zero-sized grid, which CUDA rejects with
        // "invalid configuration argument". An empty workload is a no-op -- there is no result
        // to produce.
        if (num_rays == 0) {
            return;
        }

        // Zero boxes is different: there is one answer per ray and it is "missed". Writing it
        // costs one pass and keeps the caller's buffer consistent with the documented contract
        // instead of leaving stale data that reads as hits.
        if (num_boxes == 0) {
            const int grid = static_cast<int>((num_rays + kRayBlockSize - 1) / kRayBlockSize);
            fill_missed_box_hits_kernel<<<grid, kRayBlockSize, 0, stream>>>(results, num_rays);
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        const int block_size = 256;
        const int grid_size = static_cast<int>((num_rays + block_size - 1) / block_size);
        ray_box_intersect_kernel<<<grid_size, block_size, 0, stream>>>(rays, boxes, results, num_rays, num_boxes);
        CUDA_CHECK(cudaGetLastError());
    }

    void ray_sphere_intersect_batch(const Ray* rays, const Sphere* spheres, Sphere::SphereHit* results, size_t num_rays, size_t num_spheres, cudaStream_t stream) {
        if (num_rays == 0) {
            return;
        }

        // Same contract as the box batch: no spheres means every ray missed, not "no answer".
        if (num_spheres == 0) {
            const int grid = static_cast<int>((num_rays + kRayBlockSize - 1) / kRayBlockSize);
            fill_missed_sphere_hits_kernel<<<grid, kRayBlockSize, 0, stream>>>(results, num_rays);
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        const int block_size = 256;
        const int grid_size = static_cast<int>((num_rays + block_size - 1) / block_size);
        ray_sphere_intersect_kernel<<<grid_size, block_size, 0, stream>>>(rays, spheres, results, num_rays, num_spheres);
        CUDA_CHECK(cudaGetLastError());
    }

}  // namespace cuda::raytrace
