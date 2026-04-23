#include <gtest/gtest.h>
#include "cuda/raytrace/primitives.h"

using namespace cuda::raytrace;

class RayBoxTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }

    AABB create_unit_box() {
        return AABB(Vec3(0, 0, 0), Vec3(1, 1, 1));
    }
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
