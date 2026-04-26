#ifndef NOVA_PROPERTY_TEST_HPP
#define NOVA_PROPERTY_TEST_HPP

#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <functional>
#include <chrono>

namespace nova {
namespace property {

struct TestResult {
    bool passed;
    std::string property_name;
    uint64_t seed;
    std::string failure_reason;
    size_t iterations;

    static TestResult Pass(const std::string& name, uint64_t seed, size_t iters) {
        return {true, name, seed, "", iters};
    }

    static TestResult Fail(const std::string& name, uint64_t seed,
                          const std::string& reason, size_t iters) {
        return {false, name, seed, reason, iters};
    }
};

class RandomGenerator {
public:
    explicit RandomGenerator(uint64_t seed) : rng_(seed) {}

    template<typename T>
    T Uniform(T min, T max) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng_);
    }

    template<typename T>
    T UniformFloat(T min, T max) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(rng_);
    }

    template<typename T>
    std::vector<T> Vector(size_t size, T min, T max) {
        std::vector<T> result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = UniformFloat<T>(min, max);
        }
        return result;
    }

    uint64_t GetSeed() const { return seed_; }

private:
    std::mt19937_64 rng_;
    uint64_t seed_;
};

template<typename Property>
TestResult CheckProperty(const std::string& name, Property&& prop,
                        size_t iterations = 100, uint64_t seed = 0) {
    if (seed == 0) {
        seed = std::chrono::steady_clock::now().time_since_epoch().count();
    }

    RandomGenerator gen(seed);

    for (size_t i = 0; i < iterations; ++i) {
        try {
            if (!prop(gen)) {
                return TestResult::Fail(name, seed,
                    "Property failed on iteration " + std::to_string(i), i);
            }
        } catch (const std::exception& e) {
            return TestResult::Fail(name, seed,
                std::string("Exception: ") + e.what(), i);
        }
    }

    return TestResult::Pass(name, seed, iterations);
}

#define NOVA_PROPERTY_TEST(name, iterations, generator, ...) \
    [&] { \
        auto prop = [&](auto& gen) { return (__VA_ARGS__); }; \
        return nova::property::CheckProperty(name, prop, iterations, generator); \
    }()

} // namespace property
} // namespace nova

#endif // NOVA_PROPERTY_TEST_HPP
