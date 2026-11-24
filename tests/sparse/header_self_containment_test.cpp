/**
 * @file header_self_containment_test.cpp
 * @brief Headers must compile on their own
 *
 * Compiling each of the 177 headers in include/ as the sole include of a translation unit
 * turned up four that could not:
 *
 *   cuda/async/stream_manager.h        used std::clamp with no <algorithm>
 *   cuda/benchmark/nvtx.h              used std::int64_t with no <cstdint> on the #else
 *                                      path, which is the default build
 *   cuda/distributed/sequence_parallel.h  called detail::check_comm_healthy outside the
 *                                      #if NOVA_NCCL_ENABLED block that defines detail, so
 *                                      the header did not compile without NCCL
 *   image/gaussian_blur.h              used uint8_t with no <cstdint>
 *   image/types.h                      included cuda/device/device_utils.h for the CUDA_CHECK
 *                                      macro it supplied transitively, but device_utils.h uses
 *                                      threadIdx and __syncthreads, so a host translation unit
 *                                      could not compile it at all
 *
 * Each was found by the compiler and fixed by adding the missing include or guard. The
 * headers that still do not compile under g++ use CUDA built-ins (threadIdx, blockIdx) and
 * compile under nvcc, so they are not included in the list below.
 *
 * This file includes all five, which is the regression: it did not compile before.
 */

#include <cuda/async/stream_manager.h>
#include <cuda/benchmark/nvtx.h>
#include <cuda/distributed/sequence_parallel.h>
#include <image/gaussian_blur.h>
#include <image/types.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace header_self_containment_test {

    TEST(HeaderSelfContainmentTest, TheFiveHeadersCanBeIncludedTogether) {
        // Reaching this point is the check.
        SUCCEED();
    }

    TEST(HeaderSelfContainmentTest, ImageTypesStaysAHostHeader) {
        // image/types.h is included by host translation units, so it must not depend on headers that
        // use CUDA built-ins. It used to reach CUDA_CHECK through cuda/device/device_utils.h, which
        // declares block_reduce() with threadIdx and __syncthreads inside -- constructs only nvcc
        // accepts. Including the header that actually defines the macro (cuda/device/error.h) is
        // what made this file compilable under g++ in the first place.
        //
        // If a future change re-introduces the device-only include, this .cpp stops compiling, which
        // is the whole point: this is a host translation unit.
        ImageBuffer<PixelFormat::UCHAR3> buffer(4, 4);
        EXPECT_EQ(buffer.dimensions().width, 4u);
        EXPECT_EQ(buffer.dimensions().height, 4u);
        EXPECT_EQ(buffer.dimensions().channels, 3u);
    }

    TEST(HeaderSelfContainmentTest, NvtxStubPathDeclaresItsIntegerType) {
        // The disabled build is the default, and its ScopedRange uses std::int64_t. Taking the
        // address of set_payload forces the declaration to be complete rather than merely
        // parsed.
        using SetPayload = void (cuda::benchmark::nvtx::ScopedRange::*)(std::int64_t);
        SetPayload fn = &cuda::benchmark::nvtx::ScopedRange::set_payload;
        EXPECT_NE(fn, nullptr);
    }

    TEST(HeaderSelfContainmentTest, GaussianBlurTakesAnEightBitBuffer) {
        // The signature uses uint8_t; taking its address forces the parameter types to be
        // complete rather than merely parsed. The exact arity is not the point here.
        auto fn = &cuda::algo::gaussianBlur;
        (void)fn;
        EXPECT_TRUE((std::is_pointer_v<decltype(fn)>));
    }

}  // namespace header_self_containment_test
