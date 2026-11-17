/**
 * @file buffer_thread_safety_test.cpp
 * @brief cuda::memory::Buffer has no lock, and sparse/matrix.hpp used to imply it did
 *
 * include/cuda/sparse/matrix.hpp carried "@note Thread-safe buffer operations". The buffers
 * it refers to are cuda::memory::Buffer: a raw pointer, a size, and no synchronisation.
 * resize() writes both members and frees the old pointer; copy_from() writes through the
 * pointer. Two threads writing one Buffer race, and a resize concurrent with any read can free
 * what the reader is using.
 *
 * What *is* safe is separate objects, and reading one that has stopped changing. These cases
 * pin that half — which is what the old note was reaching for — so the corrected wording has
 * something behind it.
 */

#include <cuda/memory/buffer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <thread>
#include <vector>

namespace buffer_thread_safety_test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

}  // namespace

TEST(BufferThreadSafetyTest, SeparateBuffersCanBeWrittenFromSeparateThreads) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The safe case: each thread owns its own Buffer, so there is no shared state at all.
    constexpr size_t kCount = 1024;
    std::atomic<int> failures{0};

    auto writer = [&failures] {
        cuda::memory::Buffer<float> local(kCount);
        if (local.data() == nullptr) {
            failures.fetch_add(1);
            return;
        }
        std::vector<float> host(kCount, 1.5F);
        try {
            local.copy_from(host.data(), kCount);
        } catch (...) {
            failures.fetch_add(1);
        }
    };

    std::thread first(writer);
    std::thread second(writer);
    first.join();
    second.join();

    EXPECT_EQ(failures.load(), 0) << "independent Buffers must not interfere";
}

TEST(BufferThreadSafetyTest, AFinishedBufferCanBeReadConcurrently) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The other safe case: the buffer stops changing before the readers start. data() and
    // size() are plain reads of members nothing writes any more.
    constexpr size_t kCount = 512;
    cuda::memory::Buffer<float> buffer(kCount);
    ASSERT_NE(buffer.data(), nullptr);
    std::vector<float> host(kCount, 2.0F);
    buffer.copy_from(host.data(), kCount);

    std::atomic<int> mismatches{0};
    auto reader = [&buffer, &mismatches] {
        if (buffer.size() != kCount) {
            mismatches.fetch_add(1);
        }
        if (buffer.data() == nullptr) {
            mismatches.fetch_add(1);
        }
    };

    std::thread first(reader);
    std::thread second(reader);
    first.join();
    second.join();

    EXPECT_EQ(mismatches.load(), 0) << "reads of a settled buffer are consistent";
}

TEST(BufferThreadSafetyTest, ResizeReallocatesIncludingDownToZero) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The operation that makes a Buffer unsafe to share while mutating: it frees the old
    // allocation. Checked sequentially, where it is well defined.
    cuda::memory::Buffer<float> buffer(64);
    float* first = buffer.data();
    ASSERT_NE(first, nullptr);

    buffer.resize(128);
    EXPECT_EQ(buffer.size(), 128u);
    EXPECT_NE(buffer.data(), nullptr);
    // Deliberately not asserting that the address changed: the CUDA allocator pools freed
    // blocks, so resize can hand back the pointer it just released. My first version of this
    // case asserted a new address and failed for that reason — a property of the allocator,
    // not of Buffer.

    buffer.resize(0);
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.data(), nullptr) << "a zero-size buffer holds no pointer";
}

}  // namespace buffer_thread_safety_test
