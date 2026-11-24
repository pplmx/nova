#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "cuda/algo/reduce.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

class Timer {
public:
    Timer(const char* name)
        : name_(name),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(end - start_).count();
        std::cout << std::left << std::setw(35) << name_ << std::right << std::setw(10) << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
    }

private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template <typename T>
void printResult(const char* name, T result) {
    std::cout << std::left << std::setw(35) << name << ": " << result << std::endl;
}

int main() {
    // Every device call below reports failure by throwing CudaException (Buffer, reduce_* and the
    // underlying CUDA_CHECK all do). Without a handler the process ends in std::terminate, so the
    // only thing a user sees is
    //
    //     terminate called after throwing an instance of 'cuda::device::CudaException'
    //       what(): ... CUDA error: no CUDA-capable device is detected
    //     Aborted (core dumped)
    //
    // Measured by running this binary with CUDA_VISIBLE_DEVICES="" -- after printing the banner and
    // the array size it dumped core. Catching here keeps the diagnostic, drops the core dump, and
    // returns a status a script can check.
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "   CUDA Parallel Algorithms Benchmark   " << std::endl;
        std::cout << "   (Layered Architecture)               " << std::endl;
        std::cout << "========================================" << std::endl;

        constexpr size_t N = 1 << 20;

        std::cout << "\n--- Data Setup ---" << std::endl;
        std::cout << "Array size: " << N << " elements" << std::endl;

        std::vector<int> input(N);
        std::iota(input.begin(), input.end(), 1);

        cuda::memory::Buffer<int> d_input(N);
        d_input.copy_from(input.data(), N);

        std::cout << "\n--- Reduce (Sum) ---" << std::endl;
        std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        {
            Timer t("Reduce Sum");
            int result = cuda::algo::reduce_sum(d_input.data(), N);
            printResult("  Sum result", result);
        }

        {
            Timer t("Reduce Sum Optimized");
            int result = cuda::algo::reduce_sum_optimized(d_input.data(), N);
            printResult("  Sum result", result);
        }

        {
            Timer t("Reduce Max");
            int result = cuda::algo::reduce_max(d_input.data(), N);
            printResult("  Max result", result);
        }

        {
            Timer t("Reduce Min");
            int result = cuda::algo::reduce_min(d_input.data(), N);
            printResult("  Min result", result);
        }

        std::cout << "\n--- Reduce (Buffer) ---" << std::endl;
        std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        cuda::memory::Buffer<int> d_vec(N);
        d_vec.copy_from(input.data(), input.size());

        {
            Timer t("Reduce Sum (Buffer)");
            int result = cuda::algo::reduce_sum(d_vec.data(), d_vec.size());
            printResult("  Sum result", result);
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "         Benchmark Complete!            " << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        // stderr, so it is visible even when stdout is redirected to a results file.
        std::cerr << "\nBenchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
