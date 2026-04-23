#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace cuda::raytrace {

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }

    __host__ __device__ float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return (len > 0) ? Vec3(x / len, y / len, z / len) : Vec3(0, 0, 0);
    }
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    float t_min;
    float t_max;

    __host__ __device__ Ray() : t_min(0.0f), t_max(1e20f) {}

    __host__ __device__ Ray(const Vec3& orig, const Vec3& dir)
        : origin(orig), direction(dir.normalize()), t_min(0.0f), t_max(1e20f) {}

    __host__ __device__ Vec3 point_at(float t) const {
        return Vec3(
            origin.x + direction.x * t,
            origin.y + direction.y * t,
            origin.z + direction.z * t
        );
    }
};

struct AABB {
    Vec3 min_point;
    Vec3 max_point;

    __host__ __device__ AABB() {}

    __host__ __device__ AABB(const Vec3& min_p, const Vec3& max_p)
        : min_point(min_p), max_point(max_p) {}

    __host__ __device__ AABB(const Vec3& p1, const Vec3& p2, const Vec3& p3) {
        min_point = Vec3(
            fminf(fminf(p1.x, p2.x), p3.x),
            fminf(fminf(p1.y, p2.y), p3.y),
            fminf(fminf(p1.z, p2.z), p3.z)
        );
        max_point = Vec3(
            fmaxf(fmaxf(p1.x, p2.x), p3.x),
            fmaxf(fmaxf(p1.y, p2.y), p3.y),
            fmaxf(fmaxf(p1.z, p2.z), p3.z)
        );
    }

    __host__ __device__ bool contains(const Vec3& p) const {
        return (p.x >= min_point.x && p.x <= max_point.x &&
                p.y >= min_point.y && p.y <= max_point.y &&
                p.z >= min_point.z && p.z <= max_point.z);
    }

    __host__ __device__ Vec3 center() const {
        return Vec3(
            (min_point.x + max_point.x) * 0.5f,
            (min_point.y + max_point.y) * 0.5f,
            (min_point.z + max_point.z) * 0.5f
        );
    }

    __host__ __device__ Vec3 extent() const {
        return Vec3(
            max_point.x - min_point.x,
            max_point.y - min_point.y,
            max_point.z - min_point.z
        );
    }

    __host__ __device__ void expand(const Vec3& p) {
        min_point = Vec3(fminf(min_point.x, p.x), fminf(min_point.y, p.y), fminf(min_point.z, p.z));
        max_point = Vec3(fmaxf(max_point.x, p.x), fmaxf(max_point.y, p.y), fmaxf(max_point.z, p.z));
    }

    __host__ __device__ void expand(const AABB& box) {
        expand(box.min_point);
        expand(box.max_point);
    }

    struct BoxHit {
        bool hit;
        float t_near;
        float t_far;
        Vec3 hit_normal;
    };

    __host__ __device__ BoxHit intersect(const Ray& ray) const {
        BoxHit result = {false, 0.0f, 0.0f, Vec3(0, 0, 0)};

        float tmin = ray.t_min;
        float tmax = ray.t_max;

        float inv_dir_x = (ray.direction.x != 0.0f) ? 1.0f / ray.direction.x : 1e30f;
        float inv_dir_y = (ray.direction.y != 0.0f) ? 1.0f / ray.direction.y : 1e30f;
        float inv_dir_z = (ray.direction.z != 0.0f) ? 1.0f / ray.direction.z : 1e30f;

        float t1, t2, tnear, tfar;

        t1 = (min_point.x - ray.origin.x) * inv_dir_x;
        t2 = (max_point.x - ray.origin.x) * inv_dir_x;
        tnear = fminf(t1, t2);
        tfar = fmaxf(t1, t2);
        tmin = fmaxf(tmin, tnear);
        tmax = fminf(tmax, tfar);
        if (tmin > tmax) return result;

        t1 = (min_point.y - ray.origin.y) * inv_dir_y;
        t2 = (max_point.y - ray.origin.y) * inv_dir_y;
        tnear = fminf(t1, t2);
        tfar = fmaxf(t1, t2);
        tmin = fmaxf(tmin, tnear);
        tmax = fminf(tmax, tfar);
        if (tmin > tmax) return result;

        t1 = (min_point.z - ray.origin.z) * inv_dir_z;
        t2 = (max_point.z - ray.origin.z) * inv_dir_z;
        tnear = fminf(t1, t2);
        tfar = fmaxf(t1, t2);
        tmin = fmaxf(tmin, tnear);
        tmax = fminf(tmax, tfar);
        if (tmin > tmax) return result;

        if (tmin >= ray.t_min && tmin <= ray.t_max) {
            result.hit = true;
            result.t_near = tmin;
            result.t_far = tmax;

            if (fabsf(tnear - tmin) < 1e-5f) {
                if (fabsf(inv_dir_x) < fabsf(inv_dir_y) && fabsf(inv_dir_x) < fabsf(inv_dir_z)) {
                    result.hit_normal = Vec3(t1 < t2 ? -1.0f : 1.0f, 0, 0);
                } else if (fabsf(inv_dir_y) < fabsf(inv_dir_z)) {
                    result.hit_normal = Vec3(0, t1 < t2 ? -1.0f : 1.0f, 0);
                } else {
                    result.hit_normal = Vec3(0, 0, t1 < t2 ? -1.0f : 1.0f);
                }
            }
        }

        return result;
    }
};

struct Sphere {
    Vec3 center;
    float radius;
    float radius_sq;

    __host__ __device__ Sphere() : radius(0), radius_sq(0) {}

    __host__ __device__ Sphere(const Vec3& c, float r) : center(c), radius(r), radius_sq(r * r) {}

    struct SphereHit {
        bool hit;
        float t_enter;
        float t_exit;
        Vec3 hit_normal;
        bool inside;
    };

    __host__ __device__ SphereHit intersect(const Ray& ray) const {
        SphereHit result = {false, 0.0f, 0.0f, Vec3(0, 0, 0), false};

        Vec3 oc = Vec3(
            ray.origin.x - center.x,
            ray.origin.y - center.y,
            ray.origin.z - center.z
        );

        float oc_dot_d = oc.dot(ray.direction);
        float c = oc.dot(oc) - radius_sq;
        float discriminant = oc_dot_d * oc_dot_d - c;

        if (discriminant < 0.0f) {
            return result;
        }

        float sqrt_disc = sqrtf(discriminant);
        float t = -oc_dot_d - sqrt_disc;

        if (c < 0.0f) {
            result.inside = true;
            t = -oc_dot_d + sqrt_disc;
            if (t < ray.t_min || t > ray.t_max) return result;

            result.hit = true;
            result.t_enter = 0.0f;
            result.t_exit = t;
            Vec3 hit_pt = ray.point_at(t);
            result.hit_normal = Vec3(
                (hit_pt.x - center.x) / radius,
                (hit_pt.y - center.y) / radius,
                (hit_pt.z - center.z) / radius
            );
            return result;
        }

        result.inside = false;
        if (t < ray.t_min) {
            t = -oc_dot_d + sqrt_disc;
            if (t > ray.t_max) return result;
        }

        result.hit = true;
        result.t_enter = t;
        result.t_exit = -oc_dot_d + sqrt_disc;
        Vec3 hit_pt = ray.point_at(t);
        result.hit_normal = Vec3(
            (hit_pt.x - center.x) / radius,
            (hit_pt.y - center.y) / radius,
            (hit_pt.z - center.z) / radius
        );

        return result;
    }
};

struct IntersectionStats {
    int prim_tests;
    int prim_hits;
    int bvh_nodes;

    __host__ __device__ IntersectionStats() : prim_tests(0), prim_hits(0), bvh_nodes(0) {}

    __host__ __device__ void reset() {
        prim_tests = 0;
        prim_hits = 0;
        bvh_nodes = 0;
    }

    __host__ __device__ float hit_rate() const {
        return (prim_tests > 0) ? static_cast<float>(prim_hits) / static_cast<float>(prim_tests) : 0.0f;
    }
};

}  // namespace cuda::raytrace
