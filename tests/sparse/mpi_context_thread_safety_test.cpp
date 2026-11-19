/**
 * @file mpi_context_thread_safety_test.cpp
 * @brief MpiContext's thread-safety note covers initialization, not every reader
 *
 * include/cuda/mpi/mpi_context.h said "Thread-safe singleton ensures single initialization
 * across the program". That is true and narrowly scoped: initialize() and finalize() take a
 * mutex, so two threads cannot both run MPI_Init. It does not make the class safe to read
 * while it is being written — local_rank_ is written under that mutex and read without it by
 * get_local_device_id(), which is public.
 *
 * The header now states the boundary. These cases pin what is observable without MPI present,
 * which is the singleton identity and the rank accessor's contract; the race itself is not
 * provoked, because a data race that happens to pass proves nothing.
 */

#include <cuda/mpi/mpi_context.h>

#include <gtest/gtest.h>

namespace mpi_context_thread_safety_test {

    TEST(MpiContextThreadSafetyTest, TheSingletonIsOneObjectAndStaysOneObject) {
        // Meyer's singleton: the object is created on first use and every call returns it. This is
        // the part the header's note is about at the object level.
        auto& first = cuda::mpi::MpiContext::instance();
        auto& second = cuda::mpi::MpiContext::instance();

        EXPECT_EQ(&first, &second);
    }

    TEST(MpiContextThreadSafetyTest, RepeatedInstanceCallsFromTwoThreadsReturnOneObject) {
        // Concurrent first-use is the case a singleton has to survive. If the initialisation were
        // not guarded, two threads could observe different objects here.
        cuda::mpi::MpiContext* from_first = nullptr;
        cuda::mpi::MpiContext* from_second = nullptr;

        std::thread one([&from_first] { from_first = &cuda::mpi::MpiContext::instance(); });
        std::thread two([&from_second] { from_second = &cuda::mpi::MpiContext::instance(); });
        one.join();
        two.join();

        ASSERT_NE(from_first, nullptr);
        ASSERT_NE(from_second, nullptr);
        EXPECT_EQ(from_first, from_second) << "concurrent instance() calls must agree on the object";
    }

    TEST(MpiContextThreadSafetyTest, TheRankAccessorsReportConsistentValuesAfterInitialisation) {
        // Reading after initialisation has returned is the documented-safe usage, and is where
        // these values are meaningful. Without MPI the context reports a single-rank world.
        auto& context = cuda::mpi::MpiContext::instance();

        EXPECT_GE(context.world_rank(), 0);
        EXPECT_GE(context.world_size(), 1);
        EXPECT_LE(context.world_rank(), context.world_size());
        EXPECT_GE(context.get_local_device_id(), 0) << "the device id falls back to 0 rather than an uninitialised value";
    }

}  // namespace mpi_context_thread_safety_test
