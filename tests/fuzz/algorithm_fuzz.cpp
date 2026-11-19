#include "fuzz_utils.hpp"
#include <cuda/memory/buffer.h>
#include <cuda/algo/reduce.h>
#include <parallel/scan.h>
#include <parallel/sort.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace nova {
    namespace fuzz {

        extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
            FuzzedDataProvider provider(data, size);

            // Generate operation type (0=reduce, 1=scan, 2=sort)
            auto op = provider.ConsumeIntegralInRange<int>(0, 2);

            // Limit count to prevent excessive memory usage
            auto count = provider.ConsumeIntegralInRange<size_t>(1, 4096);
            if (count > 1024) {
                count = 1024;  // Cap at reasonable size
            }

            // Generate test data
            std::vector<float> input(count);
            for (size_t i = 0; i < count; ++i) {
                input[i] = provider.ConsumeFloatingPoint<float>();
            }

            switch (op) {
                case 0: {  // reduce
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
                        return 1;  // Crash on error
                    }
                    break;
                }
                case 1: {  // scan
                    try {
                        // Calls the library. The previous body ran its own host-side prefix-sum
                        // loop, so this branch exercised the fuzz file rather than the tree.
                        //
                        // scan.h lives in cuda::algo; sort.h is in cuda::parallel. Both take
                        // memory::Buffer, which is why the data is staged on the device first.
                        cuda::memory::Buffer<float> d_in(input.size());
                        cuda::memory::Buffer<float> d_out(input.size());
                        d_in.copy_from(input.data(), input.size());
                        cuda::algo::inclusiveScan(d_in, d_out, count);

                        // A prefix sum is monotone non-decreasing only for non-negative input, so the
                        // invariant checked is the definitional one: last element equals the total.
                        std::vector<float> scanned(count);
                        d_out.copy_to(scanned.data(), count);
                        double running = 0.0;
                        for (size_t i = 0; i < count; ++i) {
                            running += static_cast<double>(input[i]);
                        }
                        const float expected = static_cast<float>(running);
                        if (!std::isfinite(expected)) {
                            return 0;  // Accumulating fuzzed data overflows; the comparison is moot.
                        }
                        if (!std::isfinite(scanned[count - 1])) {
                            return 0;
                        }
                        if (std::fabs(scanned[count - 1] - expected) > 1e-3F * std::max(1.0F, std::fabs(expected))) {
                            return 1;  // Inclusive scan disagrees with the total
                        }
                    } catch (...) {
                        return 1;
                    }
                    break;
                }
                case 2: {  // sort
                    try {
                        // Calls the library. The previous body copied the input and called std::sort,
                        // then checked the result — its own comment said it "proved only that the
                        // standard library sorts".
                        cuda::memory::Buffer<float> d_in(input.size());
                        cuda::memory::Buffer<float> d_out(input.size());
                        d_in.copy_from(input.data(), input.size());
                        cuda::parallel::bitonicSort(d_in, d_out, count);

                        std::vector<float> sorted(count);
                        d_out.copy_to(sorted.data(), count);
                        for (size_t i = 1; i < count; ++i) {
                            if (sorted[i] < sorted[i - 1]) {
                                return 1;  // Not sorted
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

    }  // namespace fuzz
}  // namespace nova
