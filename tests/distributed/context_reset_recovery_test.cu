/**
 * @file context_reset_recovery_test.cu
 * @brief A context reset must not leave library handles pointing at the old context
 *
 * execute_isolated() resets the registered singletons and then calls cudaDeviceReset(),
 * which destroys the process's CUDA context; the next runtime call builds a new one. Every
 * handle created in the old context -- cuBLAS handles, CUDA streams, NCCL communicators --
 * stops being usable, but nothing about it changes from the caller's side: the value stays
 * non-null, and the runtime answers queries about it as if it were fine.
 *
 * Measured on this host with standalone programmes:
 *
 *   cublasCreate(&h); cudaDeviceReset(); cublasDestroy(h);        // SIGSEGV in libcuda
 *   cublasCreate(&h); cudaDeviceReset(); cublasSgemm(h, ...);     // CUBLAS_STATUS_EXECUTION_FAILED
 *   cublasSetStream(fresh h, stale stream); cublasSgemm(...)      // SIGSEGV in libcuda
 *   cudaStreamSynchronize(stale); cudaStreamDestroy(stale);       // both report cudaSuccess
 *
 * The third line is what makes this a library problem rather than a test problem: a handle
 * created *after* the reset is still poisoned by a stream created before it, so caching
 * either one across a reset breaks the other. The fourth is why nothing caught it earlier --
 * the two calls that would reveal a dead stream both claim success.
 *
 * On the suite the visible symptoms were the multi-GPU matmul failing with
 * CUBLAS_STATUS_EXECUTION_FAILED when it ran after execute_isolated(), and the whole process
 * dying at exit inside cublasDestroy after reporting every test passed (exit 139).
 *
 * The fix is that the singletons owning those handles register a resetter with the test
 * isolation registry, which runs before the context is reset. These cases drive the same
 * sequence the suite does -- work, execute_isolated(), work again -- and require the second
 * round to produce the right numbers.
 *
 * The multi-GPU cases need two devices and NCCL, exactly like DistributedMatmulTest, and
 * skip for the same environmental reasons.
 */

#include <gtest/gtest.h>

#include <cuda/distributed/all_gather.h>
#include <cuda/distributed/common.h>
#include <cuda/distributed/matmul.h>
#include <cuda/memory/buffer.h>
#include <cuda/memory/buffer-inl.h>
#include <cuda/mesh/device_mesh.h>
#include <cuda/nccl/nccl_context.h>
#include <cuda/neural/matmul.h>
#include <cuda/testing/test_isolation.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

#include "distributed_test_common.h"

namespace context_reset_recovery_test {

    namespace {

        bool device_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return count > 0;
        }

        bool multigpu_nccl_ready() {
            auto& ctx = cuda::nccl::NcclContext::instance();
            try {
                ctx.initialize();
            } catch (const std::exception&) {
                return false;
            }
            return cuda::distributed::test::heal_and_ready(ctx);
        }

        /// Why a multi-GPU case cannot run here, or an empty string when it can.
        std::string multigpu_skip_reason() {
            if (!device_available()) {
                return "no CUDA device";
            }
            // device_count() is 0 until initialize() runs; the sibling matmul suite calls
            // it through multigpu_nccl_ready()/DeviceMesh below, and this gate has to ask
            // the same way or every case skips on a multi-GPU host.
            auto& mesh = cuda::mesh::DeviceMesh::instance();
            mesh.initialize();
            if (mesh.device_count() < 2) {
                return "need at least 2 GPUs";
            }
            if (std::getenv("NCCL_TESTS_AVAILABLE") == nullptr) {
                return "multi-GPU matmul requires NCCL (set NCCL_TESTS_AVAILABLE=1)";
            }
            if (!multigpu_nccl_ready()) {
                return "NCCL not enabled / no >= 2 GPUs (set NCCL_TESTS_AVAILABLE=1)";
            }
            return {};
        }

        /// Runs the same reset the suite runs between isolated cases.
        void reset_like_execute_isolated() {
            auto context = cuda::testing::TestIsolationContext::create();
            context->execute_isolated([] {});
        }

        /// One call of the genuine multi-GPU path, cross-checked against a reference
        /// computed on a single device. This is the only route to the per-device cuBLAS
        /// handle cache: the single-GPU fallback uses cuda::neural::matmul's own handle.
        void run_multigpu_matmul_and_check(int device_count, const char* when) {
            const int m = 6 * device_count - 1;  // a partial last block, as the sibling suite uses
            const int n = 16, k = 8;

            std::vector<float> h_a(static_cast<size_t>(m) * k);
            std::vector<float> h_b(static_cast<size_t>(k) * n);
            for (size_t i = 0; i < h_a.size(); ++i) {
                h_a[i] = static_cast<float>(i % 7) - 3.0f;
            }
            for (size_t i = 0; i < h_b.size(); ++i) {
                h_b[i] = static_cast<float>(i % 5) - 2.0f;
            }

            std::vector<float> h_reference(static_cast<size_t>(m) * n, 0.0f);
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < n; ++col) {
                    double sum = 0.0;
                    for (int inner = 0; inner < k; ++inner) {
                        sum += static_cast<double>(h_a[static_cast<size_t>(row) * k + inner]) * static_cast<double>(h_b[static_cast<size_t>(inner) * n + col]);
                    }
                    h_reference[static_cast<size_t>(row) * n + col] = static_cast<float>(sum);
                }
            }

            std::vector<std::vector<float>> h_results(static_cast<size_t>(device_count), std::vector<float>(static_cast<size_t>(m) * n));
            cuda::distributed::test::run_per_rank(device_count, [&](int device) {
                cuda::memory::Buffer<float> d_a(h_a.size());
                cuda::memory::Buffer<float> d_b(h_b.size());
                cuda::memory::Buffer<float> d_c(static_cast<size_t>(m) * n);
                d_a.copy_from(h_a.data(), h_a.size());
                d_b.copy_from(h_b.data(), h_b.size());

                auto result = cuda::distributed::DistributedMatmul::matmul_multi_gpu(d_a.data(), d_b.data(), d_c.data(), m, n, k);
                ASSERT_TRUE(result.ok()) << when << ": " << result.error_message;

                d_c.copy_to(h_results[static_cast<size_t>(device)].data(), h_results[static_cast<size_t>(device)].size());
            });

            for (int rank = 0; rank < device_count; ++rank) {
                SCOPED_TRACE(std::string(when) + ", rank " + std::to_string(rank));
                for (size_t i = 0; i < h_reference.size(); ++i) {
                    ASSERT_NEAR(h_results[static_cast<size_t>(rank)][i], h_reference[i], 1e-3f) << "at element " << i;
                }
            }
        }

    }  // namespace

    // The cuBLAS handles the matmul cache holds must be released before the context goes
    // away, and recreated afterwards. Before the cache registered a resetter, the second
    // matmul below handed a handle from the destroyed context to cublasSetStream, which
    // SIGSEGVs inside libcuda; the sibling suite saw CUBLAS_STATUS_EXECUTION_FAILED from it.
    TEST(ContextResetRecoveryTest, MultiGpuMatmulWorksAfterTheContextIsReset) {
        const std::string reason = multigpu_skip_reason();
        if (!reason.empty()) {
            GTEST_SKIP() << reason;
        }
        const int device_count = cuda::mesh::DeviceMesh::instance().device_count();

        run_multigpu_matmul_and_check(device_count, "before the reset");
        reset_like_execute_isolated();

        // The reset releases the NCCL context too, so it reports itself uninitialised now
        // rather than handing out communicators from the destroyed context. Re-establishing
        // it is the caller's step, and it is what the on-demand path does; a matmul without
        // it fails with "NCCL context not initialized", which is the honest answer.
        ASSERT_TRUE(multigpu_nccl_ready()) << "NCCL could not be reinitialised on the new context";

        run_multigpu_matmul_and_check(device_count, "after the reset");
    }

    // The per-device stream cache has the same requirement, and it is the harder one to
    // notice: cudaStreamSynchronize and cudaStreamDestroy both report cudaSuccess on a
    // stream from the destroyed context, so only the next cuBLAS call that binds it fails.
    TEST(ContextResetRecoveryTest, MeshStreamsAreUsableAfterTheContextIsReset) {
        const std::string reason = multigpu_skip_reason();
        if (!reason.empty()) {
            GTEST_SKIP() << reason;
        }
        auto& mesh = cuda::mesh::DeviceMesh::instance();

        auto& streams = cuda::distributed::MeshStreams::instance();
        streams.initialize(mesh.device_count());
        ASSERT_NE(streams.get_stream(0), nullptr);

        reset_like_execute_isolated();

        // The reset must have dropped the old handles, so this creates new ones on the
        // current context. Asking for the stream is what would hand back a dead one.
        streams.initialize(mesh.device_count());
        cudaStream_t stream = streams.get_stream(0);
        ASSERT_NE(stream, nullptr);
        cudaError_t err = cudaStreamSynchronize(stream);
        EXPECT_EQ(err, cudaSuccess) << "the stream handed out after the reset is not usable: " << cudaGetErrorString(err);
        (void)cudaGetLastError();
    }

    // NcclContext caches a stream and a communicator per device and answered
    // has_nccl()==true and returned the same stream pointer after a reset, so only the next
    // cuBLAS call could tell the handles were dead. It registers a resetter now too.
    TEST(ContextResetRecoveryTest, NcclContextReinitialisesAfterTheContextIsReset) {
        const std::string reason = multigpu_skip_reason();
        if (!reason.empty()) {
            GTEST_SKIP() << reason;
        }
        auto& ctx = cuda::nccl::NcclContext::instance();
        const int devices = ctx.device_count();
        ASSERT_GT(devices, 0);
        ASSERT_TRUE(ctx.has_nccl());

        reset_like_execute_isolated();

        EXPECT_FALSE(ctx.has_nccl()) << "the reset must leave NCCL uninitialised, not holding communicators from the old context";

        ctx.initialize();
        EXPECT_TRUE(ctx.has_nccl());
        EXPECT_EQ(ctx.device_count(), devices);

        cudaStream_t stream = ctx.get_stream(0);
        ASSERT_NE(stream, nullptr);
        cudaError_t err = cudaStreamSynchronize(stream);
        EXPECT_EQ(err, cudaSuccess) << "the stream handed out after reinitialising is not usable: " << cudaGetErrorString(err);
        (void)cudaGetLastError();
    }

}  // namespace context_reset_recovery_test
