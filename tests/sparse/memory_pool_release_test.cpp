/**
 * @file memory_pool_release_test.cpp
 * @brief Contracts for what MemoryPool does with a fully released block
 *
 * src/memory/memory_pool.cpp's deallocate() marks a block reusable once its offset
 * returns to zero — that is the point of pooling — but it used to clear the pointer at
 * the same time:
 *
 *     if (block.offset == 0) {
 *         block.ptr = nullptr;          // the allocation was never freed
 *         block.stream_id = -1;
 *     }
 *
 * clear() and the destructor release blocks by checking block.ptr, so nulling it made
 * the entire block unreachable and leaked it. Measured with compute-sanitizer:
 *
 *     pool that preallocated without allocating   -> 0 bytes leaked
 *     pool that allocated and fully released one  -> 4096 bytes leaked
 *     three such tests together                   -> 9216 bytes leaked
 *
 * After the fix the suite reports 0 bytes leaked in 0 allocations.
 *
 * The defect is invisible to ordinary assertions: every counter the test could read —
 * total_allocated, total_available, num_blocks — reported exactly what it should, because
 * the bookkeeping was correct and only the underlying allocation was lost. The cases
 * below pin the observable behaviour (a block is still usable after full release, and the
 * pool stops holding device memory when it is destroyed); the leak itself is only
 * visible to a tool that tracks allocations.
 */

#include <cuda/memory/memory_pool.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace cuda::memory::test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

MemoryPool::Config small_config() {
    MemoryPool::Config config;
    config.block_size = 4096;
    config.max_blocks = 2;
    config.preallocate = true;
    return config;
}

}  // namespace

// Regression: a pool that released a block leaked it entirely.
TEST(MemoryPoolReleaseTest, AFullyReleasedBlockStaysUsable) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    MemoryPool pool(small_config());
    const size_t available_before = pool.total_available();

    void* first = pool.allocate(100);
    ASSERT_NE(first, nullptr);
    pool.deallocate(first, 100);

    EXPECT_EQ(pool.total_available(), available_before)
        << "releasing must restore the pool's accounting";

    // The block must serve the next request. Before the fix it could not: its pointer
    // had been cleared, so the pool believed the block held nothing it could hand out.
    void* second = pool.allocate(100);
    ASSERT_NE(second, nullptr)
        << "the pool could not reuse the block it had just taken back";
    EXPECT_EQ(pool.total_allocated(), 100u);

    pool.deallocate(second, 100);
    EXPECT_EQ(pool.total_allocated(), 0u);
}

TEST(MemoryPoolReleaseTest, ReuseDoesNotGrowThePool) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A pool whose blocks come back intact can serve repeated requests from the same
    // allocation. A pool that lost a block's pointer on release would allocate a new
    // one each time, which is the observable symptom of the leak.
    MemoryPool pool(small_config());
    const size_t blocks_after_construction = pool.num_blocks();

    for (int i = 0; i < 20; ++i) {
        void* ptr = pool.allocate(100);
        ASSERT_NE(ptr, nullptr) << "iteration " << i;
        pool.deallocate(ptr, 100);
    }

    EXPECT_EQ(pool.num_blocks(), blocks_after_construction)
        << "repeated allocate/release cycles grew the pool instead of reusing a block";
}

// A case comparing cudaMemGetInfo before and after pool destruction was written and
// then removed. It failed under ctest -j16, and the free figure it reads is device-wide:
// any neighbouring test process allocating between the two reads moves it. Three earlier
// rounds hit the same wall on three different modules.
//
// The leak it was meant to catch is already covered without a device-wide snapshot:
// ReuseDoesNotGrowThePool fails against the pre-fix source because a pool that lost a
// block's pointer allocates a fresh one on the next request, and num_blocks() observes
// that directly. That is the right instrument — an object's own counter rather than a
// device-wide figure it cannot own.

TEST(MemoryPoolReleaseTest, DeallocatingMoreThanWasTakenIsRejected) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The deallocate path adjusts block.offset by the size it is given. Handing it a
    // pointer that was never allocated must not move the accounting.
    MemoryPool pool(small_config());
    void* ptr = pool.allocate(100);
    ASSERT_NE(ptr, nullptr);
    const size_t available = pool.total_available();

    pool.deallocate(nullptr, 100);
    EXPECT_EQ(pool.total_available(), available)
        << "deallocating null must be a no-op";

    // A pointer the pool does not know about.
    int unrelated = 0;
    pool.deallocate(&unrelated, 100);
    EXPECT_EQ(pool.total_available(), available)
        << "deallocating a foreign pointer must be a no-op";

    pool.deallocate(ptr, 100);
}

TEST(MemoryPoolReleaseTest, ZeroByteAllocationIsRefused) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    MemoryPool pool(small_config());
    const size_t available = pool.total_available();

    EXPECT_EQ(pool.allocate(0), nullptr) << "a zero-byte request has nothing to hand out";
    EXPECT_EQ(pool.total_available(), available)
        << "a refused request must not consume pool capacity";
}

TEST(MemoryPoolReleaseTest, ReleaseThenUseAcrossManyBlocks) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Exercise several blocks so the block-index bookkeeping is covered, not just the
    // single-block path.
    MemoryPool::Config config = small_config();
    config.max_blocks = 4;

    MemoryPool pool(config);
    std::vector<void*> pointers;
    for (int i = 0; i < 4; ++i) {
        void* ptr = pool.allocate(64);
        ASSERT_NE(ptr, nullptr) << "allocation " << i;
        pointers.push_back(ptr);
    }

    // Release in reverse so each block's offset returns to zero in turn.
    for (auto it = pointers.rbegin(); it != pointers.rend(); ++it) {
        pool.deallocate(*it, 64);
    }
    EXPECT_EQ(pool.total_allocated(), 0u);

    // And the pool still works afterwards.
    void* again = pool.allocate(64);
    ASSERT_NE(again, nullptr);
    pool.deallocate(again, 64);
}

}  // namespace cuda::memory::test
