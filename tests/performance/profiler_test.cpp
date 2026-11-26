#include <gtest/gtest.h>
#include <stdexcept>
#include <fstream>
#include "cuda/performance/profiler.h"

#include "cuda_alloc_injection.h"

namespace cuda::performance {

    class ProfilerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            Profiler::instance().reset();
            Profiler::instance().enable();
        }

        void TearDown() override {
            Profiler::instance().disable();
            Profiler::instance().reset();
        }
    };

    // ScopedTimer creates two events and only marks itself valid once both exist and the first is
    // recorded:
    //
    //     CUDA_CHECK(cudaEventCreate(&start_));
    //     CUDA_CHECK(cudaEventCreate(&stop_));      // <- a throw here leaves start_ owned by nobody
    //     CUDA_CHECK(cudaEventRecord(start_, 0));
    //     valid_ = true;
    //
    // Its destructor returns early when !valid_, so the failure path releases nothing. The members
    // have no default initialiser either, so even without the early return there would be no way to
    // tell which handle had been created.
    TEST_F(ProfilerTest, ScopedTimerReleasesTheEventItCreatedWhenConstructionFails) {
        // The injection only bites when there is a device to create the event on; under the
        // sanitizer build no device is visible, every creation fails for its own reason, and the
        // case would report a leak that is not there.
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "no CUDA device";
        }
        (void)cudaGetLastError();

        // Verify the injection actually reaches this code before asserting on it. --wrap redirects
        // unresolved symbols at link time, and in some build configurations the call here resolves
        // directly; when it does, fail_event_after() has no effect and this case would assert on a
        // path it never entered. The probe below is a plain cudaEventCreate from the test's own
        // translation unit and tells the two situations apart.
        cuda_test::reset_event_count();
        cuda_test::fail_event_after(0);
        cudaEvent_t probe = nullptr;
        const bool injection_works = cudaEventCreate(&probe) != cudaSuccess;
        cuda_test::stop_failing_events();
        if (probe != nullptr) {
            cudaEventDestroy(probe);
        }
        if (!injection_works) {
            GTEST_SKIP() << "cudaEventCreate injection does not reach this translation unit";
        }

        // The probe above only shows the wrapper is reachable from THIS file; ScopedTimer lives in
        // profiler.cpp and --wrap only rewrites symbols left unresolved at link time. Under -O3 the
        // library's own call can be inlined or otherwise resolved without going through the wrapper,
        // and then fail_event_after() has no effect on it even though the test's own call is
        // intercepted. That is a property of the build, not of the code under test, so the case is
        // restricted to unoptimised builds rather than made to guess.
#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__) || defined(__OPTIMIZE__)
        GTEST_SKIP() << "cudaEventCreate injection does not reach library-internal calls in this build "
                        "(sanitizer instrumentation or optimisation resolves the symbol past --wrap)";
#endif

        cuda_test::reset_event_count();

        // Let the first create through, fail the second.
        cuda_test::fail_event_after(1);
        EXPECT_THROW(ScopedTimer("kernel", Profiler::instance()), std::exception) << "the second event creation was expected to fail";
        cuda_test::stop_failing_events();

        EXPECT_EQ(cuda_test::live_event_count(), 0) << "the first event must be destroyed when the second cannot be created";
    }

    TEST_F(ProfilerTest, EnableDisable) {
        EXPECT_TRUE(Profiler::instance().is_enabled());

        Profiler::instance().disable();
        EXPECT_FALSE(Profiler::instance().is_enabled());

        Profiler::instance().enable();
        EXPECT_TRUE(Profiler::instance().is_enabled());
    }

    TEST_F(ProfilerTest, RecordKernel) {
        Profiler::instance().record_kernel("test_kernel", 1.5f);
        Profiler::instance().record_kernel("test_kernel2", 2.5f);

        auto metrics = Profiler::instance().get_kernel_metrics();
        EXPECT_EQ(metrics.size(), 2);
        EXPECT_EQ(metrics[0].name, "test_kernel");
        EXPECT_EQ(metrics[0].time_ms, 1.5f);
        EXPECT_EQ(metrics[1].name, "test_kernel2");
        EXPECT_EQ(metrics[1].time_ms, 2.5f);
    }

    TEST_F(ProfilerTest, RecordMemoryOp) {
        Profiler::instance().record_memory_op("memcpy_h2d", 1024, 0.5f);

        auto metrics = Profiler::instance().get_kernel_metrics();
        EXPECT_EQ(metrics.size(), 1);
        EXPECT_EQ(metrics[0].name, "memcpy_h2d");
        EXPECT_EQ(metrics[0].bytes_transferred, 1024);
        EXPECT_GT(metrics[0].bandwidth_gbps, 0.0f);
    }

    TEST_F(ProfilerTest, RecordCollective) {
        Profiler::instance().record_collective("all_reduce", "NCCL", 2.0f, 4);

        auto metrics = Profiler::instance().get_collective_metrics();
        EXPECT_EQ(metrics.size(), 1);
        EXPECT_EQ(metrics[0].name, "all_reduce");
        EXPECT_EQ(metrics[0].op_type, "NCCL");
        EXPECT_EQ(metrics[0].time_ms, 2.0f);
        EXPECT_EQ(metrics[0].num_ranks, 4);
    }

    TEST_F(ProfilerTest, TotalTime) {
        Profiler::instance().record_kernel("kernel1", 1.0f);
        Profiler::instance().record_kernel("kernel2", 2.0f);
        Profiler::instance().record_kernel("kernel3", 3.0f);

        EXPECT_FLOAT_EQ(Profiler::instance().get_total_kernel_time(), 6.0f);
    }

    TEST_F(ProfilerTest, Reset) {
        Profiler::instance().record_kernel("test", 1.0f);
        EXPECT_EQ(Profiler::instance().get_kernel_metrics().size(), 1);

        Profiler::instance().reset();
        EXPECT_EQ(Profiler::instance().get_kernel_metrics().size(), 0);
        EXPECT_EQ(Profiler::instance().get_collective_metrics().size(), 0);
    }

    TEST_F(ProfilerTest, ExportJson) {
        Profiler::instance().record_kernel("kernel1", 1.5f);
        Profiler::instance().record_collective("allreduce", "NCCL", 2.0f, 2);

        EXPECT_NO_THROW(Profiler::instance().export_json("/tmp/profiler_test.json"));

        std::ifstream file("/tmp/profiler_test.json");
        if (file.good()) {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();

            EXPECT_TRUE(content.find("\"kernel_metrics\"") != std::string::npos);
            EXPECT_TRUE(content.find("\"collective_metrics\"") != std::string::npos);
        }
    }

}  // namespace cuda::performance
