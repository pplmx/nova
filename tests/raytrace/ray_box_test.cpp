#include <gtest/gtest.h>
#include "cuda/raytrace/primitives.h"

using namespace cuda::raytrace;

class RayBoxTest : public ::testing::Test {
protected:
    void SetUp() override { (void)cudaDeviceSynchronize(); }

    AABB create_unit_box() { return AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)); }
};

TEST_F(RayBoxTest, RayParallelToAxisHitsBox) {
    AABB box = create_unit_box();
    Ray ray(Vec3(0.5f, 0.5f, -1.0f), Vec3(0, 0, 1));

    auto hit = box.intersect(ray);

    EXPECT_TRUE(hit.hit);
    EXPECT_GT(hit.t_near, 0.0f);
    EXPECT_LT(hit.t_near, hit.t_far);
}

TEST_F(RayBoxTest, RayMissesBoxCompletely) {
    AABB box = create_unit_box();
    Ray ray(Vec3(2.0f, 2.0f, 0.5f), Vec3(0, 0, 1));

    auto hit = box.intersect(ray);

    EXPECT_FALSE(hit.hit);
}

TEST_F(RayBoxTest, RayFromInsideBox) {
    AABB box = create_unit_box();
    Ray ray(Vec3(0.5f, 0.5f, 0.5f), Vec3(0, 0, 1));

    auto hit = box.intersect(ray);

    EXPECT_TRUE(hit.hit);
    EXPECT_LT(hit.t_near, 1.0f);
}

TEST_F(RayBoxTest, RayAtObliqueAngle) {
    AABB box = create_unit_box();
    Ray ray(Vec3(-1.0f, -1.0f, 2.0f), Vec3(1, 1, -1).normalize());

    auto hit = box.intersect(ray);

    EXPECT_TRUE(hit.hit);
    EXPECT_GT(hit.t_near, 0.0f);
    EXPECT_LT(hit.t_near, hit.t_far);
}

TEST_F(RayBoxTest, AABBConstructionFromVertices) {
    Vec3 p1(0.0f, 0.0f, 0.0f);
    Vec3 p2(1.0f, 0.0f, 0.0f);
    Vec3 p3(0.5f, 1.0f, 0.0f);

    AABB box(p1, p2, p3);

    EXPECT_EQ(box.min_point.x, 0.0f);
    EXPECT_EQ(box.max_point.x, 1.0f);
    EXPECT_EQ(box.min_point.y, 0.0f);
    EXPECT_EQ(box.max_point.y, 1.0f);
}

TEST_F(RayBoxTest, AABBContainsPoint) {
    AABB box = create_unit_box();

    EXPECT_TRUE(box.contains(Vec3(0.5f, 0.5f, 0.5f)));
    EXPECT_FALSE(box.contains(Vec3(1.5f, 0.5f, 0.5f)));
}

TEST_F(RayBoxTest, AABBCenterAndExtent) {
    AABB box(Vec3(0, 0, 0), Vec3(10, 10, 10));

    Vec3 center = box.center();
    Vec3 extent = box.extent();

    EXPECT_EQ(center.x, 5.0f);
    EXPECT_EQ(center.y, 5.0f);
    EXPECT_EQ(center.z, 5.0f);
    EXPECT_EQ(extent.x, 10.0f);
    EXPECT_EQ(extent.y, 10.0f);
    EXPECT_EQ(extent.z, 10.0f);
}

TEST_F(RayBoxTest, AABBExpandWithPoint) {
    AABB box(Vec3(0, 0, 0), Vec3(1, 1, 1));
    box.expand(Vec3(2.0f, 2.0f, 2.0f));

    EXPECT_EQ(box.max_point.x, 2.0f);
    EXPECT_EQ(box.max_point.y, 2.0f);
    EXPECT_EQ(box.max_point.z, 2.0f);
}

TEST_F(RayBoxTest, RayPointAtCorrectPosition) {
    Ray ray(Vec3(0, 0, 0), Vec3(1, 0, 0));

    Vec3 p5 = ray.point_at(5.0f);
    Vec3 p10 = ray.point_at(10.0f);

    EXPECT_EQ(p5.x, 5.0f);
    EXPECT_EQ(p10.x, 10.0f);
}

TEST_F(RayBoxTest, MultipleBoxesWithRay) {
    AABB box1(Vec3(0, 0, 0), Vec3(1, 1, 1));
    AABB box2(Vec3(2, 0, 0), Vec3(3, 1, 1));
    Ray ray(Vec3(-1, 0.5f, 0.5f), Vec3(1, 0, 0));

    auto hit1 = box1.intersect(ray);
    auto hit2 = box2.intersect(ray);

    EXPECT_TRUE(hit1.hit);
    EXPECT_TRUE(hit2.hit);
    EXPECT_LT(hit1.t_near, hit2.t_near);
}

// ---------------------------------------------------------------------------
// The hit normal was (0,0,0) for five of the six faces. It was derived from
// whichever tnear the clipping loop happened to hold after its last iteration --
// the Z slab -- and from comparing the reciprocal direction magnitudes, which
// picks the axis the ray is most nearly parallel to rather than the face it
// entered through. Nothing in this file asserted on hit_normal, so it went
// unnoticed.
//
// The convention these cases pin: hit_normal points OUT of the box and therefore
// against the incoming ray, so dot(normal, direction) < 0.
// ---------------------------------------------------------------------------

TEST_F(RayBoxTest, ReportsTheEntryFaceNormalForEveryAxis) {
    const AABB box = create_unit_box();

    struct Case {
        const char* name;
        Vec3 origin;
        Vec3 direction;
        Vec3 expected_normal;
    };
    // The box is [0,1]^3, so the min faces are at 0 and the max faces at 1.
    const Case cases[] = {
        {"entering through x = 0", Vec3(-5, 0.5f, 0.5f), Vec3(1, 0, 0), Vec3(-1, 0, 0)},
        {"entering through x = 1", Vec3(5, 0.5f, 0.5f), Vec3(-1, 0, 0), Vec3(1, 0, 0)},
        {"entering through y = 0", Vec3(0.5f, -5, 0.5f), Vec3(0, 1, 0), Vec3(0, -1, 0)},
        {"entering through y = 1", Vec3(0.5f, 5, 0.5f), Vec3(0, -1, 0), Vec3(0, 1, 0)},
        {"entering through z = 0", Vec3(0.5f, 0.5f, -5), Vec3(0, 0, 1), Vec3(0, 0, -1)},
        {"entering through z = 1", Vec3(0.5f, 0.5f, 5), Vec3(0, 0, -1), Vec3(0, 0, 1)},
    };

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        Ray ray(c.origin, c.direction);
        const auto hit = box.intersect(ray);

        ASSERT_TRUE(hit.hit);
        EXPECT_FLOAT_EQ(hit.hit_normal.x, c.expected_normal.x);
        EXPECT_FLOAT_EQ(hit.hit_normal.y, c.expected_normal.y);
        EXPECT_FLOAT_EQ(hit.hit_normal.z, c.expected_normal.z);

        // The normal must face the ray, which is the property a shading calculation relies on.
        EXPECT_LT(hit.hit_normal.dot(ray.direction), 0.0f) << "the reported normal must point against the incoming ray";
    }
}

// The normal must agree with the face the hit point actually lies on, including when the
// ray enters off-centre so that the other two slabs do not clip at t_near.
TEST_F(RayBoxTest, NormalMatchesTheFaceOfTheHitPoint) {
    const AABB box = create_unit_box();
    Ray ray(Vec3(-5, 0.3f, 0.2f), Vec3(1, 0, 0));

    const auto hit = box.intersect(ray);
    ASSERT_TRUE(hit.hit);

    const Vec3 p = ray.point_at(hit.t_near);
    EXPECT_FLOAT_EQ(p.x, box.min_point.x) << "the ray must enter through the x = 0 face";
    EXPECT_FLOAT_EQ(hit.hit_normal.x, -1.0f) << "and the normal must name that same face";
}

// A ray that starts inside was never clipped by an entry face, so there is no entry normal
// to report. That is the one case where (0,0,0) is the honest answer, and it must not be
// confused with the failures above.
TEST_F(RayBoxTest, NoEntryNormalWhenTheRayStartsInside) {
    const AABB box = create_unit_box();
    Ray ray(Vec3(0.5f, 0.5f, 0.5f), Vec3(1, 0, 0));

    const auto hit = box.intersect(ray);
    ASSERT_TRUE(hit.hit);
    EXPECT_FLOAT_EQ(hit.hit_normal.x, 0.0f);
    EXPECT_FLOAT_EQ(hit.hit_normal.y, 0.0f);
    EXPECT_FLOAT_EQ(hit.hit_normal.z, 0.0f);
}

// The clipping arithmetic must survive a ray parallel to two slabs at once: the reciprocal
// direction is clamped and the remaining slab decides.
TEST_F(RayBoxTest, ParallelToTwoSlabsStillFindsTheRemainingFace) {
    const AABB box = create_unit_box();
    Ray ray(Vec3(-5, 0.25f, 0.75f), Vec3(1, 0, 0));  // y and z direction components are 0

    const auto hit = box.intersect(ray);
    ASSERT_TRUE(hit.hit);
    EXPECT_FLOAT_EQ(hit.hit_normal.x, -1.0f);
    EXPECT_FLOAT_EQ(hit.t_near, 5.0f);
    EXPECT_FLOAT_EQ(hit.t_far, 6.0f);
}
