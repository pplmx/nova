/**
 * @file raytrace_primitives_test.cpp
 * @brief Contracts for the ray/box and ray/sphere intersection entry points
 *
 * include/cuda/raytrace/primitives.h held only the data structures; the two
 * batch entry points in src/cuda/raytrace/primitives.cu were declared nowhere.
 * The symbols were exported by the library, so they appeared to exist, while no
 * translation unit could name them. They are declared now, and these tests
 * exercise them through the public header.
 *
 * The per-primitive math (AABB::intersect, Sphere::intersect) is __host__
 * __device__, so the expected values are computed here on the host with the same
 * functions and compared against what the kernels produce. That checks the
 * kernels' selection logic rather than re-deriving geometry twice.
 */

#include <cuda/device/error.h>
#include <cuda/raytrace/primitives.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace cuda::raytrace::test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        /// RAII device array.
        template <typename T>
        class DeviceArray {
        public:
            explicit DeviceArray(const std::vector<T>& host) {
                count_ = host.size();
                if (count_ > 0) {
                    cudaMalloc(&ptr_, count_ * sizeof(T));
                    cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice);
                }
            }

            explicit DeviceArray(size_t count) {
                count_ = count;
                if (count_ > 0) {
                    cudaMalloc(&ptr_, count_ * sizeof(T));
                    cudaMemset(ptr_, 0, count_ * sizeof(T));
                }
            }

            ~DeviceArray() {
                if (ptr_ != nullptr) {
                    cudaFree(ptr_);
                }
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            const T* data() const { return ptr_; }
            T* data() { return ptr_; }

            std::vector<T> to_host() const {
                std::vector<T> host(count_);
                if (count_ > 0) {
                    cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
                }
                return host;
            }

        private:
            T* ptr_ = nullptr;
            size_t count_ = 0;
        };

    }  // namespace

    // Regression: this entry point could not be named from any other translation
    // unit because the header declared no functions at all.
    TEST(RayBoxBatchTest, IsCallableThroughThePublicHeader) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<Ray> rays = {Ray(Vec3(0.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f))};
        const std::vector<AABB> boxes = {AABB(Vec3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f))};

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<AABB> d_boxes(boxes);
        DeviceArray<AABB::BoxHit> d_results(rays.size());

        ray_box_intersect_batch(d_rays.data(), d_boxes.data(), d_results.data(), rays.size(), boxes.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        ASSERT_EQ(results.size(), 1u);
        EXPECT_TRUE(results[0].hit) << "a ray aimed at the box must hit it";
        // Origin z=-5, box face at z=-1: the near intersection is 4 units away.
        EXPECT_NEAR(results[0].t_near, 4.0f, 1e-4f);
        EXPECT_NEAR(results[0].t_far, 6.0f, 1e-4f);
    }

    TEST(RayBoxBatchTest, SelectsTheNearestOfSeveralBoxes) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<Ray> rays = {Ray(Vec3(0.0f, 0.0f, -10.0f), Vec3(0.0f, 0.0f, 1.0f))};
        // Three boxes along the ray, at increasing distance.
        const std::vector<AABB> boxes = {
            AABB(Vec3(-1.0f, -1.0f, 5.0f), Vec3(1.0f, 1.0f, 6.0f)),
            AABB(Vec3(-1.0f, -1.0f, -3.0f), Vec3(1.0f, 1.0f, -2.0f)),  // nearest
            AABB(Vec3(-1.0f, -1.0f, 1.0f), Vec3(1.0f, 1.0f, 2.0f)),
        };

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<AABB> d_boxes(boxes);
        DeviceArray<AABB::BoxHit> d_results(rays.size());

        ray_box_intersect_batch(d_rays.data(), d_boxes.data(), d_results.data(), rays.size(), boxes.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        ASSERT_TRUE(results[0].hit);
        // From z=-10 the box at z=-3 is 7 units away, closer than 11 and 15.
        EXPECT_NEAR(results[0].t_near, 7.0f, 1e-4f) << "the nearest box must win";
    }

    TEST(RayBoxBatchTest, MissesAreReported) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // A ray pointing away from the box.
        const std::vector<Ray> rays = {Ray(Vec3(0.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, -1.0f))};
        const std::vector<AABB> boxes = {AABB(Vec3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f))};

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<AABB> d_boxes(boxes);
        DeviceArray<AABB::BoxHit> d_results(rays.size());

        ray_box_intersect_batch(d_rays.data(), d_boxes.data(), d_results.data(), rays.size(), boxes.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        EXPECT_FALSE(results[0].hit) << "a ray pointing away must not report a hit";
    }

    TEST(RayBoxBatchTest, EveryRayGetsItsOwnResult) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Two rays: one hits, one does not. A kernel that shared state or indexed
        // wrongly would not reproduce this pattern.
        const std::vector<Ray> rays = {
            Ray(Vec3(0.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f)),    // hits
            Ray(Vec3(10.0f, 10.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f)),  // misses
        };
        const std::vector<AABB> boxes = {AABB(Vec3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f))};

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<AABB> d_boxes(boxes);
        DeviceArray<AABB::BoxHit> d_results(rays.size());

        ray_box_intersect_batch(d_rays.data(), d_boxes.data(), d_results.data(), rays.size(), boxes.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        ASSERT_EQ(results.size(), 2u);
        EXPECT_TRUE(results[0].hit);
        EXPECT_FALSE(results[1].hit);
    }

    // Regression: same undeclared-symbol defect as the box variant.
    TEST(RaySphereBatchTest, IsCallableThroughThePublicHeader) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<Ray> rays = {Ray(Vec3(0.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f))};
        const std::vector<Sphere> spheres = {Sphere(Vec3(0.0f, 0.0f, 0.0f), 1.0f)};

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<Sphere> d_spheres(spheres);
        DeviceArray<Sphere::SphereHit> d_results(rays.size());

        ray_sphere_intersect_batch(d_rays.data(), d_spheres.data(), d_results.data(), rays.size(), spheres.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        ASSERT_EQ(results.size(), 1u);
        EXPECT_TRUE(results[0].hit);
        // Unit sphere at the origin, ray from z=-5 along +z: enters at 4, exits at 6.
        EXPECT_NEAR(results[0].t_enter, 4.0f, 1e-4f);
        EXPECT_NEAR(results[0].t_exit, 6.0f, 1e-4f);
        EXPECT_FALSE(results[0].inside) << "the ray starts outside the sphere";
    }

    TEST(RaySphereBatchTest, SelectsTheNearestSphere) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::vector<Ray> rays = {Ray(Vec3(0.0f, 0.0f, -10.0f), Vec3(0.0f, 0.0f, 1.0f))};
        const std::vector<Sphere> spheres = {
            Sphere(Vec3(0.0f, 0.0f, 5.0f), 1.0f),
            Sphere(Vec3(0.0f, 0.0f, -3.0f), 1.0f),  // nearest
            Sphere(Vec3(0.0f, 0.0f, 1.0f), 1.0f),
        };

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<Sphere> d_spheres(spheres);
        DeviceArray<Sphere::SphereHit> d_results(rays.size());

        ray_sphere_intersect_batch(d_rays.data(), d_spheres.data(), d_results.data(), rays.size(), spheres.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        ASSERT_TRUE(results[0].hit);
        // Sphere at z=-3 with radius 1 is entered at z=-4, which is 6 units away.
        EXPECT_NEAR(results[0].t_enter, 6.0f, 1e-4f);
    }

    TEST(RaySphereBatchTest, RayStartingInsideReportsInside) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Origin is the sphere centre.
        const std::vector<Ray> rays = {Ray(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f))};
        const std::vector<Sphere> spheres = {Sphere(Vec3(0.0f, 0.0f, 0.0f), 2.0f)};

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<Sphere> d_spheres(spheres);
        DeviceArray<Sphere::SphereHit> d_results(rays.size());

        ray_sphere_intersect_batch(d_rays.data(), d_spheres.data(), d_results.data(), rays.size(), spheres.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        EXPECT_TRUE(results[0].hit);
        EXPECT_TRUE(results[0].inside) << "the ray starts inside the sphere";
        EXPECT_NEAR(results[0].t_exit, 2.0f, 1e-4f) << "it should exit at the surface";
    }

    TEST(RaySphereBatchTest, MissesAreReported) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Ray passes far to the side of the sphere.
        const std::vector<Ray> rays = {Ray(Vec3(10.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f))};
        const std::vector<Sphere> spheres = {Sphere(Vec3(0.0f, 0.0f, 0.0f), 1.0f)};

        DeviceArray<Ray> d_rays(rays);
        DeviceArray<Sphere> d_spheres(spheres);
        DeviceArray<Sphere::SphereHit> d_results(rays.size());

        ray_sphere_intersect_batch(d_rays.data(), d_spheres.data(), d_results.data(), rays.size(), spheres.size(), nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        const auto results = d_results.to_host();
        EXPECT_FALSE(results[0].hit);
    }

    TEST(RayBatchTest, EmptyBatchesAreSafe) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // Zero rays means a zero-sized grid; the launch must not fault.
        EXPECT_NO_THROW(ray_box_intersect_batch(nullptr, nullptr, nullptr, 0, 0, nullptr));
        EXPECT_NO_THROW(ray_sphere_intersect_batch(nullptr, nullptr, nullptr, 0, 0, nullptr));
        (void)cudaGetLastError();
    }

    // Zero primitives is not the same case as zero rays, and the two were handled identically:
    // both returned without writing anything. With rays present and no boxes to test against,
    // the documented answer is one "missed" result per ray, and the buffer used to keep
    // whatever the caller left in it. Measured: a results array pre-filled with
    // {hit = true, t_near = -12345} came back unchanged, so a caller reading it saw a hit that
    // no box could have produced.
    TEST(RayBatchTest, ZeroPrimitivesReportsAMissForEveryRay) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        constexpr size_t kRays = 4;

        Ray* d_rays = nullptr;
        AABB* d_boxes = nullptr;
        Sphere* d_spheres = nullptr;
        AABB::BoxHit* d_box_results = nullptr;
        Sphere::SphereHit* d_sphere_results = nullptr;
        ASSERT_EQ(cudaMalloc(&d_rays, kRays * sizeof(Ray)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_boxes, sizeof(AABB)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_spheres, sizeof(Sphere)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_box_results, kRays * sizeof(AABB::BoxHit)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_sphere_results, kRays * sizeof(Sphere::SphereHit)), cudaSuccess);

        // Poison both result arrays so "untouched" is distinguishable from "written".
        const std::vector<AABB::BoxHit> box_poison(kRays, AABB::BoxHit {true, -12345.0f, -12345.0f, Vec3(-1, -1, -1)});
        const std::vector<Sphere::SphereHit> sphere_poison(kRays, Sphere::SphereHit {true, -12345.0f, -12345.0f, Vec3(-1, -1, -1), false});
        ASSERT_EQ(cudaMemcpy(d_box_results, box_poison.data(), kRays * sizeof(AABB::BoxHit), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_sphere_results, sphere_poison.data(), kRays * sizeof(Sphere::SphereHit), cudaMemcpyHostToDevice), cudaSuccess);

        ASSERT_NO_THROW(ray_box_intersect_batch(d_rays, d_boxes, d_box_results, kRays, 0, nullptr));
        ASSERT_NO_THROW(ray_sphere_intersect_batch(d_rays, d_spheres, d_sphere_results, kRays, 0, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<AABB::BoxHit> box_out(kRays);
        std::vector<Sphere::SphereHit> sphere_out(kRays);
        ASSERT_EQ(cudaMemcpy(box_out.data(), d_box_results, kRays * sizeof(AABB::BoxHit), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(sphere_out.data(), d_sphere_results, kRays * sizeof(Sphere::SphereHit), cudaMemcpyDeviceToHost), cudaSuccess);

        for (size_t i = 0; i < kRays; ++i) {
            EXPECT_FALSE(box_out[i].hit) << "ray " << i << " cannot hit any of zero boxes, and its result still says it did";
            EXPECT_FALSE(sphere_out[i].hit) << "ray " << i << " cannot hit any of zero spheres";
        }

        cudaFree(d_rays);
        cudaFree(d_boxes);
        cudaFree(d_spheres);
        cudaFree(d_box_results);
        cudaFree(d_sphere_results);
        (void)cudaGetLastError();
    }

    TEST(IntersectionStatsTest, HitRateIsDefinedForZeroTests) {
        IntersectionStats stats;
        EXPECT_FLOAT_EQ(stats.hit_rate(), 0.0f) << "no division by zero";

        stats.prim_tests = 10;
        stats.prim_hits = 3;
        EXPECT_FLOAT_EQ(stats.hit_rate(), 0.3f);

        stats.reset();
        EXPECT_EQ(stats.prim_tests, 0);
        EXPECT_EQ(stats.prim_hits, 0);
    }

}  // namespace cuda::raytrace::test
