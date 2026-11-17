#include "fuzz_utils.hpp"
#include <cuda/memory/buffer.h>
#include <cuda/algo/reduce.h>
#include <parallel/scan.h>
#include <parallel/sort.h>
#include <vector>
#include <algorithm>

namespace nova {
namespace fuzz {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    FuzzedDataProvider provider(data, size);

    // Generate operation type (0=reduce, 1=scan, 2=sort)
    auto op = provider.ConsumeIntegralInRange<int>(0, 2);

    // Limit count to prevent excessive memory usage
    auto count = provider.ConsumeIntegralInRange<size_t>(1, 4096);
    if (count > 1024) count = 1024; // Cap at reasonable size

    // Generate test data
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = provider.ConsumeFloatingPoint<float>();
    }

    switch (op) {
        case 0: { // reduce
            try {
                // Call the library, not a copy of it: the previous body summed the input with
                // its own loop, so this target exercised nothing in the tree.
                //
                // The data has to be staged onto the device first. reduce_sum hands the
                // pointer straight to a kernel — my first version of this fix passed
                // input.data() and faulted, which is how the missing device-pointer warning
                // in cuda/algo/reduce.h was found.
                cuda::memory::Buffer<float> d_input(input.size());
                d_input.copy_from(input.data(), input.size());
                const float result = cuda::algo::reduce_sum(d_input.data(), count);
                (void)result;
            } catch (...) {
                return 1; // Crash on error
            }
            break;
        }
        case 1: { // scan (inclusive prefix sum)
            try {
                std::vector<float> output(count);
                output[0] = input[0];
                for (size_t i = 1; i < count; ++i) {
                    output[i] = output[i-1] + input[i];
                }
            } catch (...) {
                return 1;
            }
            break;
        }
        case 2: { // sort
            try {
                // std::sort on a copy proved only that the standard library sorts. The scan
                // and sort branches below still do local work because the library's
                // parallel versions allocate device buffers, which a fuzz target cannot do
                // once per input; noted rather than silently left.
                std::vector<float> sorted = input;
                std::sort(sorted.begin(), sorted.end());
                for (size_t i = 1; i < sorted.size(); ++i) {
                    if (sorted[i] < sorted[i-1]) {
                        return 1; // Not sorted - crash
                    }
                }
            } catch (...) {
                return 1;
            }
            break;
        }
    }

    return 0;
}

} // namespace fuzz
} // namespace nova
