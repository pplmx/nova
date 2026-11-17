#pragma once

/**
 * @file nvtx.h
 * @brief NVTX annotation framework with compile-time toggle for performance benchmarking
 *
 * Usage:
 *   - Enable NVTX: compile with -DNOVA_NVTX_ENABLED=1
 *   - Disable NVTX: compile with -DNOVA_NVTX_ENABLED=0 (default, zero overhead)
 *
 * The toggle uses a template parameter approach so the compiler can completely
 * eliminate dead code when disabled.
 */

#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED

#include <cstdint>
#include <nvtx3/nvtx3.hpp>

namespace cuda::benchmark::nvtx {

struct MemoryDomain {
    static constexpr char const* name{"Nova::Memory"};
};

struct DeviceDomain {
    static constexpr char const* name{"Nova::Device"};
};

struct AlgoDomain {
    static constexpr char const* name{"Nova::Algorithm"};
};

struct DistributedDomain {
    static constexpr char const* name{"Nova::Distributed"};
};

struct BenchmarkDomain {
    static constexpr char const* name{"Nova::Benchmark"};
};

template <typename Domain = BenchmarkDomain>
class ScopedRange {
public:
    explicit ScopedRange(const char* name) {
        nvtx3::mark(name);
    }

    void set_payload(int64_t payload) {
        nvtx3::mark(name_, nvtx3::payload{payload});
    }

    void set_message(const char* message) {
        nvtx3::mark(message);
    }

private:
    const char* name_;
};

template <typename Domain = BenchmarkDomain>
void mark(const char* name) {
    nvtx3::mark_in<Domain>(name);
}

template <typename Domain = BenchmarkDomain>
void mark_with_payload(const char* name, int64_t payload) {
    nvtx3::mark_in<Domain>(name, nvtx3::payload{payload});
}

inline void push_range(const char* name, uint32_t color = 0xFF88CC) {
    nvtx3::scoped_range range{name};
    (void)color;
}

inline void pop_range() {
}

}  // namespace cuda::benchmark::nvtx

#define NOVA_NVTX_SCOPED_RANGE(name) \
    cuda::benchmark::nvtx::ScopedRange<> _nova_nvtx_range_(name)

#define NOVA_NVTX_SCOPED_RANGE_DOMAIN(domain, name) \
    cuda::benchmark::nvtx::ScopedRange<cuda::benchmark::nvtx::domain> _nova_nvtx_range_(name)

#define NOVA_NVTX_MARK(name) \
    cuda::benchmark::nvtx::mark(name)

#define NOVA_NVTX_MARK_PAYLOAD(name, payload) \
    cuda::benchmark::nvtx::mark_with_payload(name, payload)

#define NOVA_NVTX_PUSH_RANGE(name) \
    cuda::benchmark::nvtx::push_range(name)

#define NOVA_NVTX_POP_RANGE() \
    cuda::benchmark::nvtx::pop_range()

#else

// The disabled build is the default one, and it uses std::int64_t below. <cstdint> was only
// included inside the enabled branch, so a translation unit that took the #else path had no
// declaration for it — found by compiling every header on its own.
#include <cstdint>

namespace cuda::benchmark::nvtx {

struct MemoryDomain {};
struct DeviceDomain {};
struct AlgoDomain {};
struct DistributedDomain {};
struct BenchmarkDomain {};

class ScopedRange {
public:
    explicit ScopedRange(const char*) {}
    void set_payload(std::int64_t) {}
    void set_message(const char*) {}
};

inline void mark(const char*) {}
inline void mark_with_payload(const char*, std::int64_t) {}
inline void push_range(const char*) {}
inline void pop_range() {}

}  // namespace cuda::benchmark::nvtx

#define NOVA_NVTX_SCOPED_RANGE(name) \
    if (false) cuda::benchmark::nvtx::ScopedRange<> _nova_nvtx_range_(name)

#define NOVA_NVTX_SCOPED_RANGE_DOMAIN(domain, name) \
    if (false) cuda::benchmark::nvtx::ScopedRange<cuda::benchmark::nvtx::domain> _nova_nvtx_range_(name)

#define NOVA_NVTX_MARK(name) \
    do { (void)sizeof(name); } while (false)

#define NOVA_NVTX_MARK_PAYLOAD(name, payload) \
    do { (void)sizeof(name); (void)sizeof(payload); } while (false)

#define NOVA_NVTX_PUSH_RANGE(name) \
    do { (void)sizeof(name); } while (false)

#define NOVA_NVTX_POP_RANGE() \
    do { } while (false)

#endif

namespace cuda::benchmark {

inline void init_nvtx() {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
    NOVA_NVTX_MARK("benchmark_start");
#endif
}

inline void finalize_nvtx() {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
    NOVA_NVTX_MARK("benchmark_end");
#endif
}

}  // namespace cuda::benchmark
