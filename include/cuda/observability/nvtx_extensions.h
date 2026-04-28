#pragma once

#include <nvtx3/nvtx3.hpp>

#ifndef NOVA_NVTX_ENABLED
#define NOVA_NVTX_ENABLED 1
#endif

#if NOVA_NVTX_ENABLED

#define NOVA_NVTX_SCOPED_RANGE(name) nvtx3::scoped_range name(name)
#define NOVA_NVTX_PUSH_RANGE(name) nvtx3::mark(name)
#define NOVA_NVTX_POP_RANGE() nvtx3::pop_range()

namespace cuda::observability {

struct NVTXDomains {
    static constexpr nvtx3::domain_handle_t Memory = nvtx3::domain_create("nova.memory");
    static constexpr nvtx3::domain_handle_t Device = nvtx3::domain_create("nova.device");
    static constexpr nvtx3::domain_handle_t Algo = nvtx3::domain_create("nova.algo");
    static constexpr nvtx3::domain_handle_t API = nvtx3::domain_create("nova.api");
    static constexpr nvtx3::domain_handle_t Production = nvtx3::domain_create("nova.production");
};

template <nvtx3::domain_handle_t Domain>
class ScopedRange {
public:
    explicit ScopedRange(const char* name) {
        nvtx3::scoped_range range(nvtx3::event<Domain>{name});
    }
};

template <nvtx3::domain_handle_t Domain>
void push_range(const char* name) {
    nvtx3::mark(nvtx3::event<Domain>{name});
}

template <nvtx3::domain_handle_t Domain>
void pop_range() {
    nvtx3::pop_range();
}

}  // namespace cuda::observability

#else

#define NOVA_NVTX_SCOPED_RANGE(name) ((void)0)
#define NOVA_NVTX_PUSH_RANGE(name) ((void)0)
#define NOVA_NVTX_POP_RANGE() ((void)0)

namespace cuda::observability {

template <typename Domain>
class ScopedRange {
public:
    explicit ScopedRange(const char*) {}
};

template <typename Domain>
void push_range(const char*) {}

template <typename Domain>
void pop_range() {}

struct NVTXDomains {
    static constexpr void* Memory = nullptr;
    static constexpr void* Device = nullptr;
    static constexpr void* Algo = nullptr;
    static constexpr void* API = nullptr;
    static constexpr void* Production = nullptr;
};

}  // namespace cuda::observability

#endif
