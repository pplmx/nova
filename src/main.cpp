#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>

#include "cuda/algo/reduce.h"
#include "cuda/api/device_vector.h"

class Timer {
public:
    Timer(const char* name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(end - start_).count();
        std::cout << std::left << std::setw(35) << name_
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3)
                  << ms << " ms" << std::endl;
    }

private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template<typename T>
void printResult(const char* name, T result) {
    std::cout << std::left << std::setw(35) << name << ": " << result << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   CUDA Parallel Algorithms Benchmark   " << std::endl;
    std::cout << "   (Layered Architecture)               " << std::endl;
    std::cout << "========================================" << std::endl;

    constexpr size_t N = 1 << 20;

    std::cout << "\n--- Data Setup ---" << std::endl;
    std::cout << "Array size: " << N << " elements" << std::endl;

    std::vector<int> input(N);
    std::iota(input.begin(), input.end(), 1);

    int *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "\n--- Reduce (Sum) ---" << std::endl;
    std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    int result = 0;
    {
        Timer t("Reduce Sum");
        result = cuda::algo::reduce_sum(d_input, N);
    }
    printResult("  Sum result", result);

    {
        Timer t("Reduce Sum Optimized");
        result = cuda::algo::reduce_sum_optimized(d_input, N);
    }
    printResult("  Sum result", result);

    int maxResult = 0;
    {
        Timer t("Reduce Max");
        maxResult = cuda::algo::reduce_max(d_input, N);
    }
    printResult("  Max result", maxResult);

    int minResult = 0;
    {
        Timer t("Reduce Min");
        minResult = cuda::algo::reduce_min(d_input, N);
    }
    printResult("  Min result", minResult);

    std::cout << "\n--- Reduce (DeviceVector) ---" << std::endl;
    std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    cuda::api::DeviceVector<int> d_vec(N);
    d_vec.copy_from(input);

    {
        Timer t("Reduce Sum (DeviceVector)");
        result = cuda::algo::reduce_sum(d_vec.data(), d_vec.size());
    }
    printResult("  Sum result", result);

    CUDA_CHECK(cudaFree(d_input));

    std::cout << "\n========================================" << std::endl;
    std::cout << "         Benchmark Complete!            " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
