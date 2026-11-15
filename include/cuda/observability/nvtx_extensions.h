#pragma once

#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED

#include <nvtx3/nvtx3.hpp>

// nvtx3 has no pop_range() and no explicit push/pop pair in the unversioned
// API: `mark` is an instantaneous marker, not the start of a range, and the
// explicit pair is start_range()/end_range() around a range_handle. Mapping
// PUSH_RANGE onto mark() therefore emitted a point event where a range was
// intended, and POP_RANGE onto nvtx3::pop_range() did not compile at all.
//
// The macros wrap the push/pop pair in a scope so callers keep the simple
// PUSH/POP form. Because a range handle must outlive the pair, the handle lives
// in a thread-local stack.
#define NOVA_NVTX_SCOPED_RANGE(name) nvtx3::scoped_range name{#name}
#define NOVA_NVTX_PUSH_RANGE(name) \
    ::cuda::observability::detail::push_range_impl(name)
#define NOVA_NVTX_POP_RANGE() ::cuda::observability::detail::pop_range_impl()

namespace cuda::observability {

/**
 * @brief NVTX domain identity
 *
 * Nine identical nullptr values could not distinguish anything, so each domain
 * now has its own address. The value is an opaque tag, not an nvtxDomainHandle_t:
 * a real domain must be created with nvtxDomainCreateA and destroyed with
 * nvtxDomainDestroy, and owning that lifetime is outside this header.
 */
using NVTXDomainHandle = const void*;

namespace detail {

inline NVTXDomainHandle domain_tag(char& storage) {
    return &storage;
}

// Small fixed stack of in-flight range handles: NVTX ranges nest, so PUSH/POP
// must too.
inline constexpr int kMaxNestedRanges = 64;
inline thread_local nvtx3::range_handle g_range_stack[kMaxNestedRanges];
inline thread_local int g_range_depth = 0;

inline void push_range_impl(const char* name) {
    if (name == nullptr || g_range_depth >= kMaxNestedRanges) {
        return;
    }
    g_range_stack[g_range_depth++] = nvtx3::start_range(name);
}

inline void pop_range_impl() {
    if (g_range_depth <= 0) {
        return;
    }
    nvtx3::end_range(g_range_stack[--g_range_depth]);
}

}  // namespace detail

struct NVTXDomains {
    static inline char Memory_tag = 0;
    static inline char Device_tag = 0;
    static inline char Algo_tag = 0;
    static inline char API_tag = 0;
    static inline char Production_tag = 0;
    static inline char Performance_tag = 0;
    static inline char NVBlox_tag = 0;
    static inline char Fusion_tag = 0;
    static inline char Bandwidth_tag = 0;

    static constexpr NVTXDomainHandle Memory = &Memory_tag;
    static constexpr NVTXDomainHandle Device = &Device_tag;
    static constexpr NVTXDomainHandle Algo = &Algo_tag;
    static constexpr NVTXDomainHandle API = &API_tag;
    static constexpr NVTXDomainHandle Production = &Production_tag;
    static constexpr NVTXDomainHandle Performance = &Performance_tag;
    static constexpr NVTXDomainHandle NVBlox = &NVBlox_tag;
    static constexpr NVTXDomainHandle Fusion = &Fusion_tag;
    static constexpr NVTXDomainHandle Bandwidth = &Bandwidth_tag;
};

/**
 * @brief RAII range covering a scope
 *
 * `Domain` is a value of type NVTXDomainHandle in BOTH configurations. The
 * disabled branch used to take a TYPE here, so a call site written for one build
 * failed to compile against the other.
 */
template <NVTXDomainHandle Domain>
class ScopedRange {
public:
    explicit ScopedRange(const char* name) : range_(name) {}

private:
    nvtx3::scoped_range range_;
};

template <NVTXDomainHandle Domain>
void push_range(const char* name) {
    detail::push_range_impl(name);
}

template <NVTXDomainHandle Domain>
void pop_range() {
    detail::pop_range_impl();
}

}  // namespace cuda::observability

#else

// The scope form must still DECLARE the named object in this configuration: the
// enabled build expands it to a variable declaration, so expanding to a
// statement here made every call site that used the name afterwards fail to
// compile. The disabled object is an empty type with the same lifetime.
#define NOVA_NVTX_SCOPED_RANGE(name) \
    ::cuda::observability::NullScopedRange name{}
#define NOVA_NVTX_PUSH_RANGE(name) ((void)0)
#define NOVA_NVTX_POP_RANGE() ((void)0)

#define NOVA_NVTX_PUSH_RANGE(name) ((void)0)
#define NOVA_NVTX_POP_RANGE() ((void)0)

namespace cuda::observability {

// Same shape as the enabled build: NVTXDomainHandle is a value, so ScopedRange
// and the range helpers take a non-type template parameter in both branches.
using NVTXDomainHandle = const void*;

/// Placeholder object for NOVA_NVTX_SCOPED_RANGE when NVTX is disabled.
struct NullScopedRange {
    NullScopedRange() noexcept = default;
    explicit NullScopedRange(const char*) noexcept {}
};

struct NVTXDomains {
    static inline char Memory_tag = 0;
    static inline char Device_tag = 0;
    static inline char Algo_tag = 0;
    static inline char API_tag = 0;
    static inline char Production_tag = 0;
    static inline char Performance_tag = 0;
    static inline char NVBlox_tag = 0;
    static inline char Fusion_tag = 0;
    static inline char Bandwidth_tag = 0;

    static constexpr NVTXDomainHandle Memory = &Memory_tag;
    static constexpr NVTXDomainHandle Device = &Device_tag;
    static constexpr NVTXDomainHandle Algo = &Algo_tag;
    static constexpr NVTXDomainHandle API = &API_tag;
    static constexpr NVTXDomainHandle Production = &Production_tag;
    static constexpr NVTXDomainHandle Performance = &Performance_tag;
    static constexpr NVTXDomainHandle NVBlox = &NVBlox_tag;
    static constexpr NVTXDomainHandle Fusion = &Fusion_tag;
    static constexpr NVTXDomainHandle Bandwidth = &Bandwidth_tag;
};

template <NVTXDomainHandle Domain>
class ScopedRange {
public:
    explicit ScopedRange(const char*) {}
};

template <NVTXDomainHandle Domain>
void push_range(const char*) {}

template <NVTXDomainHandle Domain>
void pop_range() {}

}  // namespace cuda::observability

#endif
