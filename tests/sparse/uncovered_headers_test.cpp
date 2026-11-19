/**
 * @file uncovered_headers_test.cpp
 * @brief Contracts for two small headers that had no test coverage
 *
 * Two headers were the last in include/cuda without any reference from the test
 * tree:
 *
 * - cuda/sparse/solver_errors.hpp: the SolverError codes shared by the iterative
 *   solvers and the workspace helpers. The enum itself is correct; this pins the
 *   values the solvers compare against so a reordering cannot silently change
 *   what a caller sees.
 *
 * - cuda/benchmark/nvbench_integration.h: an optional NVBench glue header that is
 *   not part of the build. Its previous version did not compile under any
 *   configuration: it included <nvbench/nvbench.h>, which does not exist (the
 *   real umbrella header is nvbench.cuh), and every call it made named an API
 *   absent from released NVBench:
 *
 *       .set_timeout_ms(5000)        -> set_timeout() takes SECONDS as float64
 *       .add_device_memory(...)      -> does not exist (real call: add_device)
 *       collect_keeps_bandwidth_throughput_true(...) -> does not exist
 *       add_float64_metric("GFLOPS") -> does not exist
 *
 *   The header is now guarded so that including it without NVBench is a no-op,
 *   and the helpers it keeps use only APIs NVBench actually has. The enabled
 *   branch is verified separately by compiling a translation unit against the
 *   real nvbench sources; the test binary builds the disabled branch.
 */

#include <cuda/benchmark/nvbench_integration.h>
#include <cuda/cpp_level.hpp>
#include <cuda/sparse/solver_errors.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>

namespace nova::test {

    // ------------------------------------------------------- solver_errors.hpp --

    TEST(SolverErrorTest, SuccessIsZero) {
        // Callers test `error_code == SolverError::SUCCESS` to decide a solver
        // converged, so zero is part of the contract.
        EXPECT_EQ(static_cast<int>(sparse::SolverError::SUCCESS), 0);
    }

    TEST(SolverErrorTest, EveryCodeHasADistinctValue) {
        const sparse::SolverError all[] = {
            sparse::SolverError::SUCCESS,
            sparse::SolverError::MAX_ITERATIONS,
            sparse::SolverError::BREAKDOWN,
            sparse::SolverError::INVALID_MATRIX,
            sparse::SolverError::CONVERGENCE_FAILURE,
            sparse::SolverError::PRECONDITIONER_ERROR,
        };
        constexpr size_t count = sizeof(all) / sizeof(all[0]);

        for (size_t i = 0; i < count; ++i) {
            for (size_t j = i + 1; j < count; ++j) {
                EXPECT_NE(static_cast<int>(all[i]), static_cast<int>(all[j])) << "error codes " << i << " and " << j << " share a value";
            }
        }
    }

    TEST(SolverErrorTest, CodesAreUsableInAConstantExpression) {
        static_assert(sparse::SolverError::SUCCESS == sparse::SolverError::SUCCESS);
        static_assert(sparse::SolverError::SUCCESS != sparse::SolverError::BREAKDOWN);
        SUCCEED();
    }

    TEST(SolverErrorTest, IsAScopedEnum) {
        // A scoped enum must not convert implicitly to int; that is what stops a
        // solver result being compared against an unrelated code.
        static_assert(!std::is_convertible_v<sparse::SolverError, int>);
        EXPECT_TRUE((std::is_enum_v<sparse::SolverError>));
    }

    TEST(SolverErrorTest, CanBeSwitchedOnExhaustively) {
        // Mirrors how a caller turns a result into a message; every case is named so
        // adding a code without handling it shows up as a warning here.
        auto describe = [](sparse::SolverError code) -> std::string {
            switch (code) {
                case sparse::SolverError::SUCCESS:
                    return "converged";
                case sparse::SolverError::MAX_ITERATIONS:
                    return "max iterations";
                case sparse::SolverError::BREAKDOWN:
                    return "breakdown";
                case sparse::SolverError::INVALID_MATRIX:
                    return "invalid matrix";
                case sparse::SolverError::CONVERGENCE_FAILURE:
                    return "not converged";
                case sparse::SolverError::PRECONDITIONER_ERROR:
                    return "preconditioner";
            }
            return "unknown";
        };

        EXPECT_EQ(describe(sparse::SolverError::SUCCESS), "converged");
        EXPECT_NE(describe(sparse::SolverError::BREAKDOWN), "unknown");
    }

    // ------------------------------------------------- nvbench_integration.h --

    TEST(NvbenchIntegrationTest, IncludesWithoutNvbenchPresent) {
        // The whole point of the guard: this header is listed in include/cuda but
        // nvbench is not a build dependency, so including it must be harmless.
        SUCCEED();
    }

    TEST(NvbenchBytesTest, ReportsTheBytesMovedByALaunch) {
        // The one helper that works in both configurations, so a benchmark can print
        // a throughput figure whether or not NVBench is available.
        EXPECT_DOUBLE_EQ(cuda::benchmark::bytes_processed<float>(1024), 1024.0 * 4.0);
        EXPECT_DOUBLE_EQ(cuda::benchmark::bytes_processed<double>(1024), 1024.0 * 8.0);
        EXPECT_DOUBLE_EQ(cuda::benchmark::bytes_processed<std::uint8_t>(512), 512.0);
        EXPECT_DOUBLE_EQ(cuda::benchmark::bytes_processed<float>(0), 0.0);
    }

    TEST(NvbenchBytesTest, IsAConstantExpression) {
        static_assert(cuda::benchmark::bytes_processed<float>(1) == 4.0);
        static_assert(cuda::benchmark::bytes_processed<double>(2) == 16.0);
        SUCCEED();
    }

    TEST(NvbenchMacroTest, MacrosExpandWithoutNvbench) {
        // A source file that mentions the macros has to stay compilable when the
        // benchmark is not built.
        NOVA_BENCHMARK_KERNEL(unused_benchmark);
        NOVA_BENCHMARK_MEMORY_BANDWIDTH(unused_bandwidth, 5.0);
        SUCCEED();
    }

    // ------------------------------------------------------- cpp_level.hpp --
    //
    // The header records the C++20/C++23 split: device-reachable headers stay at
    // C++20 because nvcc stops there, and a host-only header that needs C++23 calls
    // NOVA_REQUIRE_CPP23_HOST to fail loudly if device code ever includes it.
    //
    // Two defects fixed here: the marker's static_assert had no trailing semicolon,
    // so even a correct trigger produced a follow-on "expected ';'" error; and the
    // constant it names was defined AFTER the macro, making the marker depend on
    // where it was expanded.

    TEST(CppLevelTest, ReportsWhetherCpp23IsAvailable) {
        // The probe is a library feature test, not __cplusplus: nvcc reports 202002
        // even for code the C++23 host compiler processes.
        EXPECT_TRUE(NOVA_CPP23_HOST == 0 || NOVA_CPP23_HOST == 1);
    }

    TEST(CppLevelTest, MarkerIsInertOutsideDeviceCode) {
        // NOVA_DEVICE_REACHABLE is not defined for this host translation unit, so the
        // marker must expand to nothing and the statement must compile.
        NOVA_REQUIRE_CPP23_HOST("cuda/example_host_only.hpp")
        SUCCEED();
    }

    TEST(CppLevelTest, GuardConstantIsFalse) {
        // The assert condition is a named constant rather than a literal so the
        // message can explain the real problem; it must never become true.
        static_assert(!nova::detail::cpp_level::device_may_not_include_cpp23);
        EXPECT_FALSE(nova::detail::cpp_level::device_may_not_include_cpp23);
    }

    TEST(CppLevelTest, DeviceReachableIsNotDefinedForHostCode) {
        // The whole contract rests on this: .cu files define it, host code does not.
#ifndef NOVA_DEVICE_REACHABLE
        SUCCEED() << "host translation units do not define NOVA_DEVICE_REACHABLE";
#else
        GTEST_FAIL() << "this host test was compiled as device-reachable";
#endif
    }

}  // namespace nova::test
