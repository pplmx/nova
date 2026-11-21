#include <gtest/gtest.h>
#include "cuda/raytrace/primitives.h"

using namespace cuda::raytrace;

class RaySphereTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }

    Sphere create_unit_sphere() { return Sphere(Vec3(0, 0, 0), 1.0f); }
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

// ---------------------------------------------------------------------------
// A ray that starts outside a sphere and points AWAY from it reported hit = true
// with t_enter = -4 on a unit sphere at the origin. Both roots are behind the
// origin, and the code only checked the far root against t_max, never against
// t_min -- so a renderer received a hit behind the camera.
// ---------------------------------------------------------------------------

TEST_F(RaySphereTest, RayPointingAwayFromTheSphereMisses) {
    const Sphere sphere(Vec3(0, 0, 0), 1.0f);
    Ray ray(Vec3(5, 0, 0), Vec3(1, 0, 0));  // starts outside, heads further away

    const auto hit = sphere.intersect(ray);

    EXPECT_FALSE(hit.hit) << "both roots are behind the origin, so there is no hit ahead of the ray";
}

// The contract the failure above broke: any reported t must lie within the ray's own range.
TEST_F(RaySphereTest, AnyReportedHitLiesAheadOfTheOrigin) {
    const Sphere sphere(Vec3(0, 0, 0), 1.0f);

    for (const Vec3 direction : {Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0.6f, 0.8f, 0)}) {
        for (const float distance : {5.0f, 2.0f, 1.5f, 0.5f}) {
            Ray ray(direction * distance, direction);
            const auto hit = sphere.intersect(ray);
            if (hit.hit) {
                EXPECT_GE(hit.t_enter, ray.t_min) << "origin at " << distance << " along its own direction";
                EXPECT_GE(hit.t_exit, ray.t_min);
            }
        }
    }
}

// The ordinary outside case must still report both roots, in order.
TEST_F(RaySphereTest, OutsideRayReportsBothRoots) {
    const Sphere sphere(Vec3(0, 0, 0), 1.0f);
    Ray ray(Vec3(-5, 0, 0), Vec3(1, 0, 0));

    const auto hit = sphere.intersect(ray);
    ASSERT_TRUE(hit.hit);
    EXPECT_FLOAT_EQ(hit.t_enter, 4.0f);
    EXPECT_FLOAT_EQ(hit.t_exit, 6.0f);
    EXPECT_FALSE(hit.inside);
}

// A ray starting inside reports enter = 0 (it is already there) and the exit ahead.
TEST_F(RaySphereTest, InsideRayStartsAtZeroAndReportsTheExit) {
    const Sphere sphere(Vec3(0, 0, 0), 1.0f);
    Ray ray(Vec3(0, 0, 0), Vec3(1, 0, 0));

    const auto hit = sphere.intersect(ray);
    ASSERT_TRUE(hit.hit);
    EXPECT_TRUE(hit.inside);
    EXPECT_FLOAT_EQ(hit.t_enter, 0.0f);
    EXPECT_FLOAT_EQ(hit.t_exit, 1.0f);
}

// A grazing ray touches at exactly one point, so both roots coincide.
TEST_F(RaySphereTest, TangentRayReportsASingleContact) {
    const Sphere sphere(Vec3(0, 0, 0), 1.0f);
    Ray ray(Vec3(-5, 1, 0), Vec3(1, 0, 0));

    const auto hit = sphere.intersect(ray);
    ASSERT_TRUE(hit.hit);
    EXPECT_FLOAT_EQ(hit.t_enter, 5.0f);
    EXPECT_FLOAT_EQ(hit.t_exit, 5.0f);
}
