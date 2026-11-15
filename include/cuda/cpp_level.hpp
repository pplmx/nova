/**
 * @file cpp_level.hpp
 * @brief Single place that records which C++ standard each header may target
 *
 * The project does not compile with one standard. Host code is C++23
 * (CMAKE_CXX_STANDARD 23), but nvcc 12.9 tops out at -std=c++20, so everything
 * a .cu file can reach has to stay C++20. Before this header that split was
 * implicit: nothing stopped a C++23-only construct from being added to a header
 * that device code includes, and the failure then surfaced in some unrelated
 * .cu file long after the change.
 *
 * The contract:
 *
 * 1. A .cu translation unit defines NOVA_DEVICE_REACHABLE (wired in
 *    CMakeLists.txt for the CUDA language).
 * 2. A host-only header that uses C++23 library facilities calls
 *    NOVA_REQUIRE_CPP23_HOST("<header name>") at namespace scope, right after
 *    its includes.
 *
 * Then including such a header from device code stops the build with a message
 * naming it, instead of a confusing failure deep inside a template. This header
 * itself is included by everything, so it must NOT be the thing that fires —
 * only the marker call does.
 *
 * @note Stays C++17-clean and pulls in no other nova header. It does include
 *       <version> so the feature-test macros are visible.
 */

#ifndef NOVA_CUDA_CPP_LEVEL_HPP
#define NOVA_CUDA_CPP_LEVEL_HPP

#include <version>

// __cplusplus is not a reliable "has C++23" signal here: g++ 13 in -std=c++23
// mode reports 202100, and nvcc reports 202002 even for code the C++23 host
// compiler will process. Test for a C++23 library feature instead, which is
// what the host-only headers actually need.
#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#define NOVA_CPP23_HOST 1
#else
#define NOVA_CPP23_HOST 0
#endif

// NOVA_REQUIRE_CPP23_HOST(name) expands to a static_assert whose condition is
// always false when device code reaches it. The condition is written in terms
// of a macro, so it fires at definition time rather than on instantiation.
#if defined(NOVA_DEVICE_REACHABLE) && !NOVA_CPP23_HOST
#define NOVA_REQUIRE_CPP23_HOST(header_name)                                    \
    static_assert(::nova::detail::cpp_level::device_may_not_include_cpp23,      \
                  header_name " requires host C++23 but was included from "     \
                              "device code. nvcc stops at C++20; keep this "    \
                              "header out of the .cu include graph.")
#else
#define NOVA_REQUIRE_CPP23_HOST(header_name)
#endif

namespace nova::detail::cpp_level {
/** Never true: the marker above names the real problem in its message. */
inline constexpr bool device_may_not_include_cpp23 = false;
}  // namespace nova::detail::cpp_level

#endif  // NOVA_CUDA_CPP_LEVEL_HPP
