// BROKEN - do not wire it to a libFuzzer driver without fixing it first.
//
// cudaMalloc returns a device pointer and the body then calls memset() on it from the host:
//
//     cudaMalloc(&ptr, alloc_size);
//     memset(ptr, 0x42, std::min(alloc_size, size_t{256}));
//
// A host memset through a device pointer faults. This is the same mistake MemoryPoisonGuard had
// (fixed in src/testing/memory_safety.cpp, commit 1e403a1): the pointee is device memory, so a
// host write or scan has to be staged through cudaMemcpy. Connecting this target would make the
// fuzzer report a crash that belongs to the harness, not to MemoryPool.
//
// It also does not use MemoryPool at all — it allocates with raw cudaMalloc and calls the result
// "fuzzing a memory pool". The include of cuda/memory/memory_pool.h is unused by the body.
//
// Left in place rather than deleted because removing files is the maintainer's call. See
// tests/fuzz/README_STATUS.md.

#include "fuzz_utils.hpp"
#include <cuda/memory/memory_pool.h>
#include <cuda_runtime.h>
#include <cstring>

namespace nova {
    namespace fuzz {

        extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
            FuzzedDataProvider provider(data, size);

            auto alloc_size = provider.ConsumeIntegralInRange<size_t>(1, 1024 * 1024);
            auto alignment = provider.ConsumeIntegralInRange<size_t>(1, 256);

            // Clamp alignment to valid CUDA alignment values
            if (alignment < 1) {
                alignment = 1;
            }
            if (alignment > 256) {
                alignment = 256;
            }

            // Create a simple memory pool for fuzzing
            // Note: In production, we'd use the full MemoryPool implementation
            // For fuzzing, we test basic allocation patterns
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, alloc_size);

            if (err == cudaSuccess && ptr != nullptr) {
                // Touch the memory to ensure it's accessible
                memset(ptr, 0x42, std::min(alloc_size, size_t{256}));

                err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    return 1;  // Crash on free error
                }
            }

            return 0;
        }

    }  // namespace fuzz
}  // namespace nova
