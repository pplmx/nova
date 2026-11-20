/**
 * @file memory_pool_fuzz.cpp
 * @brief libFuzzer target for cuda::memory::MemoryPool
 *
 * Fuzzes the pool the way a caller would use it: a fuzzed sequence of allocations,
 * frees, defragmentation and metric reads against a pool whose configuration is
 * also taken from the input.
 *
 * History, because both earlier versions looked like fuzz targets and were not:
 *
 * 1. It allocated with raw cudaMalloc and then called host memset() on the result,
 *    which is a device pointer. That faulted on every input, so the target crashed
 *    in its own first statement and never reached the pool at all.
 * 2. Even with that fixed it "did not use MemoryPool at all" -- its own header said
 *    so. It exercised the CUDA runtime and called the result fuzzing a memory pool.
 *
 * What it checks now is the pool's own invariants after every operation, which is
 * what a fuzzer is for: a pool that returns overlapping blocks, loses a block it
 * was told to free, or reports metrics that disagree with its own counters is a
 * defect the ordinary tests only catch where someone thought to look.
 */

#include "fuzz_utils.hpp"

#include <cuda/memory/memory_pool.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nova {
    namespace fuzz {

        namespace {

            /// Bytes the harness writes into every block it hands out, so two live
            /// allocations that share storage are detected rather than merely suspected.
            constexpr size_t kTagBytes = 64;

            /// Fill the first kTagBytes of a device block with a recognisable pattern.
            /// Device memory needs a device write; a host memset through a device
            /// pointer is the fault that made the first version of this file useless.
            bool tag_block(void* ptr, size_t bytes, unsigned char tag) {
                const size_t n = std::min(bytes, kTagBytes);
                if (n == 0) {
                    return true;
                }
                return cudaMemset(ptr, tag, n) == cudaSuccess;
            }

            /// Read back the tag byte previously written at the start of the block.
            /// Returns false when the read itself fails.
            bool read_tag(void* ptr, unsigned char& out) {
                return cudaMemcpy(&out, ptr, sizeof(out), cudaMemcpyDeviceToHost) == cudaSuccess;
            }

        }  // namespace

        extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, size_t size) {
            FuzzedDataProvider provider(data, size);

            // Pool configuration from the input. Kept small: this is a correctness
            // target, and a pool that tries to preallocate gigabytes would make the
            // fuzzer measure the device instead of the code.
            cuda::memory::MemoryPool::Config config;
            config.block_size = provider.ConsumeIntegralInRange<size_t>(1, 64 * 1024);
            config.max_blocks = static_cast<size_t>(provider.ConsumeIntegralInRange<int>(1, 8));
            config.preallocate = false;  // lazy: the harness drives when memory is touched

            cuda::memory::MemoryPool pool(config);
            // A failed allocation is a result, not a crash: record it rather than throw.
            pool.set_throw_on_failure(false);

            // Live allocations: pointer -> (size, tag). The tag distinguishes blocks so
            // a block handed out twice is visible as a wrong tag, not just a stale value.
            std::unordered_map<void*, std::pair<size_t, unsigned char>> live;
            unsigned char next_tag = 1;

            const int ops = provider.ConsumeIntegralInRange<int>(1, 64);
            for (int i = 0; i < ops; ++i) {
                if (size == 0) {
                    break;
                }
                const bool do_alloc = provider.ConsumeIntegralInRange<int>(0, 1) == 1;

                if (do_alloc || live.empty()) {
                    // An allocation of 0 bytes is a legitimate edge case; the pool must
                    // reject it or return something usable, not corrupt itself.
                    const size_t bytes = provider.ConsumeIntegralInRange<size_t>(0, 4096);
                    void* ptr = pool.allocate(bytes <= 0 ? 1 : bytes);
                    if (ptr == nullptr) {
                        continue;  // exhausted: a valid outcome
                    }

                    // Every live block must have distinct storage. If the pool hands out
                    // a pointer that is already live, tagging it would be overwritten on
                    // the next read and the check below catches it.
                    if (live.count(ptr) != 0) {
                        return 1;  // the same block was handed out twice without a free
                    }

                    if (!tag_block(ptr, bytes, next_tag)) {
                        (void)cudaGetLastError();
                        return 1;  // the block we were given is not writable
                    }
                    live.emplace(ptr, std::make_pair(bytes <= 0 ? 1 : bytes, next_tag));
                    ++next_tag;
                    if (next_tag == 0) {
                        next_tag = 1;
                    }
                } else {
                    // Free a random live block, and verify the others survived: a pool
                    // that frees neighbouring storage would change their contents.
                    const size_t pick = provider.ConsumeIntegralInRange<size_t>(0, live.size() - 1);
                    auto it = live.begin();
                    std::advance(it, static_cast<long>(std::min(pick, live.size() - 1)));
                    void* freed = it->first;
                    const size_t freed_bytes = it->second.first;
                    live.erase(it);
                    pool.deallocate(freed, freed_bytes);
                }

                // Invariant after every operation: each live block still holds its tag.
                for (const auto& [ptr, info] : live) {
                    unsigned char got = 0;
                    if (!read_tag(ptr, got)) {
                        (void)cudaGetLastError();
                        return 1;
                    }
                    if (got != info.second) {
                        return 1;  // a live block's contents changed: overlapping storage
                    }
                }

                // Metrics must not report more live allocations than the pool made.
                const auto metrics = pool.get_metrics();
                if (metrics.peak_allocated_bytes == 0 && !live.empty()) {
                    return 1;  // live blocks but no recorded peak
                }
                if (pool.num_allocations() > 0 && live.empty()) {
                    // The harness freed everything; the pool still thinks it holds blocks.
                    return 1;
                }
            }

            // Defragment with live blocks outstanding, then re-check them: collapsing
            // fragmentation must move contents, and this is the only place that is
            // exercised on data the harness still remembers.
            pool.defragment();
            for (const auto& [ptr, info] : live) {
                unsigned char got = 0;
                if (!read_tag(ptr, got)) {
                    (void)cudaGetLastError();
                    return 1;
                }
                if (got != info.second) {
                    return 1;  // defragment lost or overwrote a live block
                }
            }

            // Release everything: a pool that cannot return to zero live allocations has
            // leaked a mapping even if the memory itself is still reachable.
            for (const auto& [ptr, info] : live) {
                pool.deallocate(ptr, info.first);
            }
            live.clear();
            if (pool.num_allocations() != 0) {
                return 1;  // blocks remain after every allocation was freed
            }

            (void)cudaGetLastError();
            return 0;
        }

    }  // namespace fuzz
}  // namespace nova
