/**
 * @file scan_race_test.cpp
 * @brief Contracts for the optimised exhaustive-scan kernel's shared-memory accesses
 *
 * src/parallel/scan.cu implements a Hillis-Steele inclusive scan. The optimised
 * variant fused the read and the write of each step into one guarded block:
 *
 *     for (int offset = 1; offset < size; offset *= 2) {
 *         if (tid >= offset) {
 *             temp[tid] += temp[tid - offset];      // read and write, no barrier
 *         }
 *         __syncthreads();
 *     }
 *
 * Thread `tid` reads temp[tid - offset] while thread `tid - offset` may be writing it
 * in the same step. compute-sanitizer --tool racecheck reports it:
 *
 *     Race reported between Read access at
 *     exclusiveScanOptimizedKernelImpl<int, 2>(...)+0x170
 *
 * (2 hazards). The loop now reads into a register, synchronises, then writes.
 *
 * Stated plainly: the results were already correct before the fix, because the value
 * a thread needs comes from the previous step as long as the hardware schedules the
 * read first, and it usually does. The cases below therefore pass against both
 * versions — they guard the arithmetic against a future change, not the race. Only
 * racecheck can see the defect, and it reports 0 hazards afterwards.
 */

#include <parallel/scan.h>

#include <cuda/memory/buffer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <numeric>
#include <vector>

namespace scan_race_test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

/// Owns a device int array.
class DeviceArray {
public:
    explicit DeviceArray(const std::vector<int>& host) : count_(host.size()) {
        cudaMalloc(&ptr_, (count_ > 0 ? count_ : 1) * sizeof(int));
        if (count_ > 0) {
            cudaMemcpy(ptr_, host.data(), count_ * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    explicit DeviceArray(size_t count) : count_(count) {
        cudaMalloc(&ptr_, (count_ > 0 ? count_ : 1) * sizeof(int));
        cudaMemset(ptr_, 0, (count_ > 0 ? count_ : 1) * sizeof(int));
    }

    ~DeviceArray() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    int* data() { return ptr_; }

    std::vector<int> read() const {
        std::vector<int> host(count_, 0);
        if (count_ > 0) {
            cudaMemcpy(host.data(), ptr_, count_ * sizeof(int), cudaMemcpyDeviceToHost);
        }
        return host;
    }

private:
    int* ptr_ = nullptr;
    size_t count_ = 0;
};

/// Runs the optimised scan through the Buffer-based API and returns the result.
std::vector<int> run_optimized(const std::vector<int>& input) {
    cuda::memory::Buffer<int> d_input(input.size());
    cuda::memory::Buffer<int> d_output(input.size());
    d_input.copy_from(input.data(), input.size());
    cuda::algo::exclusiveScanOptimized(d_input, d_output, input.size());
    cudaDeviceSynchronize();
    std::vector<int> result(input.size(), 0);
    d_output.copy_to(result.data(), input.size());
    return result;
}

/// Runs the simple scan through the same API, for the consistency case.
std::vector<int> run_simple(const std::vector<int>& input) {
    cuda::memory::Buffer<int> d_input(input.size());
    cuda::memory::Buffer<int> d_output(input.size());
    d_input.copy_from(input.data(), input.size());
    cuda::algo::exclusiveScan(d_input, d_output, input.size());
    cudaDeviceSynchronize();
    std::vector<int> result(input.size(), 0);
    d_output.copy_to(result.data(), input.size());
    return result;
}

/// The exclusive prefix sum the kernel is supposed to compute.
std::vector<int> expected_exclusive_scan(const std::vector<int>& input) {
    std::vector<int> result(input.size(), 0);
    int running = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = running;
        running += input[i];
    }
    return result;
}

}  // namespace

TEST(ScanRaceTest, OptimisedScanMatchesTheDefinitionOverAPowerOfTwo) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<int> input = {3, 1, 4, 1, 5, 9, 2, 6};
    EXPECT_EQ(run_optimized(input), expected_exclusive_scan(input));
}

TEST(ScanRaceTest, OptimisedScanMatchesTheDefinitionOverANonPowerOfTwo) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // A size the doubling loop cannot divide evenly: the last step is partial, which
    // is exactly where an off-by-one in the offset guard shows up.
    for (const size_t size : {size_t{5}, size_t{7}, size_t{13}, size_t{31}}) {
        std::vector<int> input(size);
        for (size_t i = 0; i < size; ++i) {
            input[i] = static_cast<int>((i * 3) % 7) + 1;
        }

        EXPECT_EQ(run_optimized(input), expected_exclusive_scan(input))
            << "size " << size;
    }
}

TEST(ScanRaceTest, OptimisedAndSimpleScansAgree) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The two implementations use different loops; they must produce the same answer
    // for every size sampled.
    for (const size_t size : {size_t{1}, size_t{2}, size_t{8}, size_t{64}, size_t{255}}) {
        std::vector<int> input(size);
        for (size_t i = 0; i < size; ++i) {
            input[i] = static_cast<int>(i % 5) + 1;
        }

        EXPECT_EQ(run_optimized(input), run_simple(input)) << "size " << size;
    }
}

TEST(ScanRaceTest, ASingleElementScansToZero) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // The smallest input: nothing precedes the only element, so its exclusive prefix
    // sum is zero. A guard that assumed at least one doubling step would break here.
    const std::vector<int> input = {42};
    const auto result = run_optimized(input);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0);
}

TEST(ScanRaceTest, AllOnesProducesTheIndexSequence) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // With every input equal to 1 the exclusive scan is 0,1,2,3,...: any element
    // duplicated or dropped by the tree shows immediately.
    constexpr size_t size = 100;
    const std::vector<int> input(size, 1);
    const auto result = run_optimized(input);
    ASSERT_EQ(result.size(), size);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i)) << "element " << i;
    }
}

TEST(ScanRaceTest, LargeValuesDoNotWrap) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // Every element is large but the total stays inside int, so a correct scan never
    // wraps. Values this size make a misplaced accumulation obvious.
    constexpr size_t size = 32;
    const std::vector<int> input(size, 1000000);
    EXPECT_EQ(run_optimized(input), expected_exclusive_scan(input));
}

}  // namespace scan_race_test
