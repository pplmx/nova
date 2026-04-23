#include <gtest/gtest.h>
#include "cuda/raytrace/primitives.h"

using namespace cuda::raytrace;

class RaySphereTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }

    Sphere create_unit_sphere() {
        return Sphere(Vec3(0, 0, 0), 1.0f);
    }
};

TEST_F(RaySphereTest, RayMissesSphere) {
    Sphere sphere = create_unit_sphere();
    Ray ray(Vec3(0, 0, -5), Vec3(1, 0, 0));

    auto hit = sphere.intersect(ray);

    EXPECT_FALSE(hit.hit);
}

TEST_F(RaySphereTest, RayHitsSphereFromOutside) {
    Sphere sphere = create_unit_sphere();
    Ray ray(Vec3(0, 0, -5), Vec3(0, 0, 1));

    auto hit = sphere.intersect(ray);

    EXPECT_TRUE(hit.hit);
    EXPECT_FALSE(hit.inside);
    EXPECT_GT(hit.t_enter, 0.0f);
    EXPECT_GT(hit.t_exit, hit.t_enter);
}

TEST_F(RaySphereTest, RayFromInsideSphere) {
    Sphere sphere = create_unit_sphere();
    Ray ray(Vec3(0, 0, 0), Vec3(0, 0, 1));

    auto hit = sphere.intersect(ray);

    EXPECT_TRUE(hit.hit);
    EXPECT_TRUE(hit.inside);
    EXPECT_EQ(hit.t_enter, 0.0f);
}

TEST_F(RaySphereTest, GrazingHitTangent) {
    Sphere sphere = create_unit_sphere();
    Ray ray(Vec3(1.0f, 0, -1.0f), Vec3(0, 0, 1));

    auto hit = sphere.intersect(ray);

    EXPECT_TRUE(hit.hit);
}

TEST_F(RaySphereTest, NormalPointsOutwardFromCenter) {
    Sphere sphere(Vec3(0, 0, 0), 1.0f);
    Ray ray(Vec3(0, 0, -3), Vec3(0, 0, 1));

    auto hit = sphere.intersect(ray);

    EXPECT_TRUE(hit.hit);
}

TEST_F(RaySphereTest, MultipleRaysSameSphere) {
    Sphere sphere = create_unit_sphere();

    Ray ray1(Vec3(0, 0, -5), Vec3(0, 0, 1));
    Ray ray2(Vec3(5, 0, 0), Vec3(-1, 0, 0));
    Ray ray3(Vec3(0, 5, 0), Vec3(0, -1, 0));

    auto hit1 = sphere.intersect(ray1);
    auto hit2 = sphere.intersect(ray2);
    auto hit3 = sphere.intersect(ray3);

    EXPECT_TRUE(hit1.hit);
    EXPECT_TRUE(hit2.hit);
    EXPECT_TRUE(hit3.hit);
}

TEST_F(RaySphereTest, SphereConstruction) {
    Sphere sphere(Vec3(5.0f, 10.0f, 15.0f), 2.5f);

    EXPECT_EQ(sphere.center.x, 5.0f);
    EXPECT_EQ(sphere.center.y, 10.0f);
    EXPECT_EQ(sphere.center.z, 15.0f);
    EXPECT_EQ(sphere.radius, 2.5f);
    EXPECT_EQ(sphere.radius_sq, 6.25f);
}

TEST_F(RaySphereTest, RayBehindSphere) {
    Sphere sphere = create_unit_sphere();
    Ray ray(Vec3(0, 0, 5), Vec3(0, 0, -1));

    auto hit = sphere.intersect(ray);

    EXPECT_TRUE(hit.hit);
    EXPECT_FALSE(hit.inside);
}
