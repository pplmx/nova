/**
 * @file fuzz_target_compiles_test.cpp
 * @brief Keep algorithm_fuzz.cpp compiling and its three branches running
 *
 * tests/fuzz/algorithm_fuzz.cpp is a libFuzzer entry point, so nothing in the normal build
 * compiles it. That is how its scan and sort branches came to run host-side loops instead of
 * calling the library: a file nobody compiles cannot rot visibly.
 *
 * Including the .cpp here does two things. It keeps the file compiling as part of every build,
 * and the case below drives LLVMFuzzerTestOneInput directly, so all three branches execute in
 * the ordinary test run. Both branches were rewritten to call cuda::algo::inclusiveScan and
 * cuda::parallel::bitonicSort after this file was found to be exercising only itself.
 *
 * This is not a substitute for fuzzing, which needs a libFuzzer driver and a clang build.
 *
 * LIMITATION, measured rather than assumed: these cases pass against both the library-calling
 * version and the earlier host-loop version of the scan and sort branches. They prove the file
 * compiles and that the entry point runs all three branches without crashing; they cannot tell
 * whether a branch reaches into the library. I checked that by stashing the rewrite and
 * re-running — both passed. What fixes that distinction is code review of the branch bodies,
 * not a test, and the branches now visibly call cuda::algo::inclusiveScan and
 * cuda::parallel::bitonicSort.
 */

#include "../fuzz/algorithm_fuzz.cpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

namespace fuzz_target_compiles_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        // FuzzedDataProvider consumes the op selector from the front of the buffer, so a leading byte
        // with a known value steers the branch. The rest is pseudo-random but deterministic.
        std::vector<unsigned char> input_for(unsigned char op) {
            std::vector<unsigned char> buffer(256);
            for (size_t i = 0; i < buffer.size(); ++i) {
                buffer[i] = static_cast<unsigned char>((i * 37 + 11) & 0xFF);
            }
            buffer[0] = op;
            return buffer;
        }

    }  // namespace

    TEST(FuzzTargetCompilesTest, AllThreeBranchesRunWithoutCrashing) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // 0 = reduce, 1 = scan, 2 = sort; each stages data onto the device and calls the library.
        for (unsigned char op = 0; op < 3; ++op) {
            const auto buffer = input_for(op);
            EXPECT_EQ(nova::fuzz::LLVMFuzzerTestOneInput(buffer.data(), buffer.size()), 0) << "branch " << static_cast<int>(op) << " reported a failure";
        }

        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "the context must survive all three";
    }

    TEST(FuzzTargetCompilesTest, AnEmptyInputIsHandled) {
        if (!device_available()) {
            GTEST_SKIP() << "no CUDA device";
        }

        // FuzzedDataProvider gets nothing to consume; the entry point must still return cleanly
        // rather than reading past the buffer.
        const unsigned char byte = 0;
        EXPECT_EQ(nova::fuzz::LLVMFuzzerTestOneInput(&byte, 0), 0);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }

}  // namespace fuzz_target_compiles_test
