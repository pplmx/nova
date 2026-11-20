#include "fuzz_utils.hpp"
#include <cuda/memory/buffer.h>
#include <cuda/algo/reduce.h>
#include <parallel/scan.h>
#include <parallel/sort.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace nova {
    namespace fuzz {

        namespace {
            /// Report a violated invariant.
            ///
            /// libFuzzer ignores LLVMFuzzerTestOneInput's return value: only a signal or
            /// an explicit abort is a finding. Every check below used to `return 1` with
            /// a comment saying "Crash on error", so a disagreement between the library
            /// and its own definition was counted as a successful run. Same defect as
            /// memory_pool_fuzz.cpp had.
            [[noreturn]] void fail(const char* what) {
                std::fprintf(stderr, "algorithm_fuzz: invariant violated: %s\n", what);
                std::abort();
            }
        }  // namespace

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
                        fail("the library threw on fuzzed input");
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
                            fail("inclusive scan disagrees with the total");
                        }
                    } catch (...) {
                        fail("the library threw on fuzzed input");
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

                        // Ordered is not sufficient. A sort that writes nothing leaves
                        // the output buffer as it allocated it -- zeroed -- and an
                        // all-zero vector is ordered, so the check below passed a
                        // bitonicSort whose final copy had been deleted (measured:
                        // sorted[0] stayed 0.0 for a descending 9-element input). Two
                        // properties together are what "sorted" means:
                        //
                        //   ordered, and a permutation of the input.
                        //
                        // NaN defeats both comparisons, so an input containing one is
                        // skipped rather than reported: the library's NaN policy is a
                        // separate contract, not this target's subject.
                        for (const float v : input) {
                            if (!std::isfinite(v)) {
                                return 0;
                            }
                        }
                        for (size_t i = 1; i < count; ++i) {
                            if (sorted[i] < sorted[i - 1]) {
                                fail("bitonicSort output is not ordered");
                            }
                        }
                        std::vector<float> want = input;
                        std::sort(want.begin(), want.end());
                        for (size_t i = 0; i < count; ++i) {
                            if (sorted[i] != want[i]) {
                                fail("bitonicSort output is not a permutation of its input");
                            }
                        }
                    } catch (...) {
                        fail("the library threw on fuzzed input");
                    }
                    break;
                }
            }

            return 0;
        }

    }  // namespace fuzz
}  // namespace nova
