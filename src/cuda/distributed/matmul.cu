/**
 * @file matmul.cu
 * @brief Multi-GPU distributed matrix multiply implementation
 *
 * Row-wise split: each GPU computes rows [r*rows_per_gpu, min((r+1)*rows_per_gpu, m))
 * of C, then an NCCL all-gather reconstructs the full result on every device.
 *
 * Single-GPU fallback (MGPU-13): Direct delegation to cuda::neural::matmul.
 *
 * Multi-GPU path (MGPU-12): DistributedMatmul::matmul_multi_gpu implements the
 * real row-split compute + NCclAllGather. It is driven thread-per-rank (one
 * thread per device pinned with cudaSetDevice), exactly like the NCCL
 * collective suite — genuine multi-device NCCL does not require separate OS
 * processes. See decision-matmul-thread-per-rank.
 */

#include "cuda/distributed/matmul.h"
#include "cuda/distributed/barrier.h"
#include "cuda/distributed/common.h"
#include "cuda/mesh/device_mesh.h"
#include "cuda/mesh/peer_copy.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"
#include "cuda/nccl/nccl_all_gather.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/neural/matmul.h"
#include "cuda/device/error.h"
#include "cuda/testing/test_isolation.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

namespace cuda::distributed {

    namespace {

        /**
         * @brief Thread-safe cache of one cuBLAS handle per device
         *
         * Rank threads pin to different devices on the multi-GPU path and may request
         * their per-device handle concurrently for the first time, so the lookups and
         * appends are serialized. Handles are created under a ScopedDevice, which binds
         * each one to its own device's context.
         *
         * A cuBLAS handle is a child of the CUDA *context* that was current when it was
         * created, not of the process, and it stops being usable when that context goes
         * away -- cudaDeviceReset() here, a driver-driven context loss elsewhere. The
         * handle value stays non-null and the runtime offers no way to tell it is dead:
         *
         *     cublasCreate(&h);                      // ok
         *     cudaDeviceReset();                     // h now points at a dead context
         *     cublasDestroy(h);                      // SIGSEGV inside libcuda
         *     cublasSgemm(h, ...);                   // CUBLAS_STATUS_EXECUTION_FAILED
         *
         * Measured on this host with a standalone programme, and on the suite both ways:
         * a multi-GPU matmul running after execute_isolated() failed with
         * CUBLAS_STATUS_EXECUTION_FAILED, and the process then died at exit inside
         * cublasDestroy after reporting every test passed (exit 139).
         *
         * Two things follow, and the previous version had neither:
         *
         *  - The handles must be destroyed while their context is still alive. The
         *    reset registry does that: it runs from reset_all_singletons(), which
         *    execute_isolated() calls BEFORE cudaDeviceReset() precisely so singletons
         *    can release their handles first (see docs/ci.md). release() is the hook.
         *  - The cache must not be destroyed by a static destructor. That runs inside
         *    __run_exit_handlers, which is exactly where cublasDestroy faulted, so the
         *    object is intentionally leaked and the driver reclaims it at exit -- the
         *    same trade MeshStreams and NcclContext make.
         */
        struct CublasHandleCache {
            std::vector<cublasHandle_t> handles;
            std::vector<int> devices;
            std::mutex mutex;

            /// Destroy every cached handle on the context it belongs to. Must run while
            /// those contexts are alive, which is why the reset registry calls it.
            void release() {
                std::lock_guard<std::mutex> lock(mutex);
                for (size_t i = 0; i < handles.size(); ++i) {
                    if (handles[i] == nullptr) {
                        continue;
                    }
                    (void)cudaSetDevice(devices[i]);
                    (void)cublasDestroy(handles[i]);
                }
                handles.clear();
                devices.clear();
                (void)cudaGetLastError();
            }
        };

        /// Process-wide and intentionally never destroyed; see CublasHandleCache.
        CublasHandleCache& cublas_cache() {
            static CublasHandleCache* cache = new CublasHandleCache();
            return *cache;
        }

        cublasHandle_t get_cublas_handle_for_device(int device) {
            CublasHandleCache& cache = cublas_cache();

            {
                std::lock_guard<std::mutex> lock(cache.mutex);
                for (size_t i = 0; i < cache.devices.size(); ++i) {
                    if (cache.devices[i] == device) {
                        return cache.handles[i];
                    }
                }
            }

            cuda::mesh::ScopedDevice guard(device);
            cublasHandle_t handle;
            CUBLAS_CHECK(cublasCreate(&handle));

            {
                std::lock_guard<std::mutex> lock(cache.mutex);
                for (size_t i = 0; i < cache.devices.size(); ++i) {
                    if (cache.devices[i] == device) {
                        // Another thread created it while we were out of the lock.
                        cublasDestroy(handle);
                        return cache.handles[i];
                    }
                }
                cache.handles.push_back(handle);
                cache.devices.push_back(device);
                return handle;
            }
        }

        /// Releases the cached handles when the test harness resets the CUDA context.
        /// Registered here rather than in test_isolation.cpp so the registration sits
        /// next to the state it clears, which is the convention documented on
        /// register_singleton_resetter().
        struct CublasCacheRegistration {
            CublasCacheRegistration() {
                cuda::testing::register_singleton_resetter("CublasHandleCache", [] { cublas_cache().release(); });
            }
        } g_cublas_cache_registration;

    }  // anonymous namespace

    bool DistributedMatmul::needs_multi_gpu() {
        auto& mesh = cuda::mesh::DeviceMesh::instance();
        mesh.initialize();
        return mesh.device_count() > 1;
    }

    void DistributedMatmul::matmul(const float* A, const float* B, float* C, int m, int n, int k, DistributedMatmulOptions options) {
        matmul_async(A, B, C, m, n, k, 0, options);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void DistributedMatmul::matmul_async(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream, DistributedMatmulOptions options) {
        auto& mesh = cuda::mesh::DeviceMesh::instance();
        mesh.initialize();

        // Single-GPU fallback (MGPU-13)
        // Always use single-GPU path for correctness in single-process execution
        // Multi-GPU operation requires proper multi-process execution
        //
        // The stream now reaches cuBLAS. It used to be ignored -- the parameter was commented out
        // in the signature -- so the work always went to the null stream regardless of what the
        // caller passed. Measured before: cudaStreamQuery(passed_stream) returned cudaSuccess
        // immediately after submission, i.e. nothing had been queued there.
        matmul_single_gpu(A, B, C, m, n, k, options, stream);
    }

    void DistributedMatmul::matmul_single_gpu(const float* A, const float* B, float* C, int m, int n, int k, DistributedMatmulOptions options, cudaStream_t stream) {
        cuda::neural::MatmulOptions neural_opts;
        neural_opts.stream = stream;
        neural_opts.alpha = options.alpha;
        neural_opts.beta = options.beta;
        neural_opts.trans_a = options.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        neural_opts.trans_b = options.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

        cuda::neural::matmul(A, B, C, m, n, k, neural_opts);
    }

    cuda::nccl::NcclResult DistributedMatmul::matmul_multi_gpu(const float* A, const float* B, float* C, int m, int n, int k, DistributedMatmulOptions options) {
        auto& ctx = cuda::nccl::NcclContext::instance();
        if (!ctx.has_nccl()) {
            return cuda::nccl::NcclResult {
                .code = ncclInternalError,
                .error_message = "NCCL context not initialized for multi-GPU matmul (call NcclContext::instance().initialize() from a thread-per-rank context)"};
        }

        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            return cuda::nccl::NcclResult {.code = ncclSystemError, .error_message = cudaGetErrorString(err)};
        }

        const int n_dev = ctx.device_count();

        // Single-rank fallback: this rank is the whole group, so the result is a
        // plain local matmul. Preserves the needs_multi_gpu()==false semantics
        // while keeping NCCL-collective behaviour out of the single-GPU path.
        // trans_a/trans_b are honored on this path (full-matrix, no row slicing).
        if (n_dev <= 1) {
            try {
                matmul_single_gpu(A, B, C, m, n, k, options);
            } catch (const std::exception& e) {
                return cuda::nccl::NcclResult {.code = ncclInternalError, .error_message = e.what()};
            }
            return cuda::nccl::NcclResult {};
        }

        // Fail fast on inputs the row-split path cannot express, before allocating
        // anything or entering the collective, rather than emit a silently wrong
        // result:
        //  - transposed operands: A's row index would move into the column
        //    dimension and no longer compose with row slicing;
        //  - an empty-or-negative last slice: with rows_per_gpu = ceil(m / n_dev),
        //    the last rank's first row is (n_dev-1)*rows_per_gpu, so when that is
        //    >= m its slice is empty or negative and its seed copy would underflow
        //    to a huge size (and its A offset read past the matrix). This covers
        //    both m < n_dev (e.g. m=3, n_dev=4) and the sliver just above a
        //    multiple of n_dev (e.g. m=5, n_dev=4 -> rows=2, 3*2=6 >= 5).
        //
        // NOTE (collective contract): every error return below happens BEFORE the
        // NCCL all-gather. Under the thread-per-rank contract, that rank then fails
        // to post its collective while the healthy ranks block in safe_nccl_call's
        // timeout and eventually ncclCommAbort the process-wide communicators,
        // poisoning NcclContext for all later NCCL users. The inputs validated here
        // are rank-independent (all ranks share m/n/k/options), so a correctly
        // driven call rejects identically on every rank and never hits that path;
        // per-rank failures (has_nccl, cudaGetDevice, rank_of_device, allocation,
        // post-GEMM) are environmental and are the documented contract risk (see
        // matmul_multi_gpu's docstring).
        if (options.trans_a || options.trans_b) {
            return cuda::nccl::NcclResult {.code = ncclInvalidArgument, .error_message = "transposed operands are not supported by the distributed row-split path"};
        }
        const int rows_per_gpu = (m + n_dev - 1) / n_dev;
        if (m < n_dev || (n_dev - 1) * rows_per_gpu >= m) {
            return cuda::nccl::NcclResult {.code = ncclInvalidArgument,
                                           .error_message = "m (" + std::to_string(m) + ") leaves device count (" + std::to_string(n_dev) + ") a rank with no rows to compute: "
                                                            + "rows_per_gpu = ceil(m/n_dev) = " + std::to_string(rows_per_gpu) + " does not give every rank at least one row"};
        }

        // Row-wise split: rank r owns rows [r*rows_per_gpu, min((r+1)*rows_per_gpu, m))
        // of C, except the last rank which stops at m. Each rank computes its
        // local_m inner rows locally, then ncclAllGather concatenates the padded
        // equal-size blocks.
        //
        // The NCCL rank is the position of the current device in the group's
        // configured device list, NOT the raw device index: they coincide only for
        // the default mesh-init group {0..n-1}, whereas a custom NcclContextConfig
        // (e.g. {0, 2, 5} or a reordered set) diverges them. rank_of_device() also
        // throws if the current device is not in the group at all — fail fast here
        // instead of computing a misattributed row slice on a device outside the set.
        int rank = 0;
        try {
            rank = ctx.rank_of_device(device);
        } catch (const std::exception& e) {
            return cuda::nccl::NcclResult {.code = ncclInvalidArgument, .error_message = e.what()};
        }

        const int start_row = rank * rows_per_gpu;
        const int local_m = std::min(rows_per_gpu, m - start_row);
        const size_t local_count = static_cast<size_t>(rows_per_gpu) * n;
        const size_t total_count = static_cast<size_t>(n_dev) * local_count;

        try {
            cuda::memory::Buffer<float> local(local_count);
            cuda::memory::Buffer<float> gathered(total_count);

            // Each rank's contribution: rows [start_row, start_row+local_m) of C,
            // i.e. the leading local_m rows (local_m x n) of the padded block.
            // The tail of the block past local_m*n is uninitialized; all-gather
            // carries it, but the caller only reads the first m*n of the gathered
            // buffer, which always land on the real rows (rows_per_gpu >= local_m,
            // and only the last rank's block is ever short).
            // Seed the local dense block with this rank's rows of the caller's C so
            // that beta accumulation matches the single-GPU semantics (cuBLAS reads
            // C's pre-call contents for the beta*C term). Only the first local_m*n
            // entries are the real rows; the padding tail is whatever the fresh
            // allocation holds and is ignored after all-gather.
            cudaError_t seed_err = cudaMemcpy(local.data(), C + static_cast<size_t>(start_row) * n, static_cast<size_t>(local_m) * n * sizeof(float), cudaMemcpyDeviceToDevice);
            if (seed_err != cudaSuccess) {
                return cuda::nccl::NcclResult {.code = ncclSystemError, .error_message = cudaGetErrorString(seed_err)};
            }

            // Same per-device stream for the GEMM and the all-gather: the GEMM must
            // be ordered before the gather (which reads `local`). Pinning both to
            // the context's per-device stream makes the ordering explicit and
            // stream-capture safe, instead of relying on the legacy default stream
            // synchronizing implicitly with every other stream.
            cudaStream_t stream = ctx.get_stream(device);

            // The cuBLAS handle cache is keyed on the CUDA device index and binds
            // each handle to that device's context, so the current device — not the
            // NCCL rank — is the lookup key. rank != device for non-default groups.
            cuda::neural::MatmulOptions neural_opts;
            neural_opts.handle = get_cublas_handle_for_device(device);
            neural_opts.alpha = options.alpha;
            neural_opts.beta = options.beta;
            neural_opts.trans_a = CUBLAS_OP_N;
            neural_opts.trans_b = CUBLAS_OP_N;
            CUBLAS_CHECK(cublasSetStream(neural_opts.handle, stream));
            cuda::neural::matmul(A + static_cast<size_t>(start_row) * k, B, local.data(), local_m, n, k, neural_opts);

            cuda::nccl::NcclAllGather gather(ctx);
            cuda::nccl::NcclResult result = gather.all_gather(local.data(), gathered.data(), static_cast<size_t>(rows_per_gpu) * n, ncclFloat32, stream);
            if (!result.ok()) {
                return result;
            }

            cudaError_t cp_err = cudaMemcpy(C, gathered.data(), static_cast<size_t>(m) * n * sizeof(float), cudaMemcpyDeviceToDevice);
            if (cp_err != cudaSuccess) {
                return cuda::nccl::NcclResult {.code = ncclSystemError, .error_message = cudaGetErrorString(cp_err)};
            }
            return cuda::nccl::NcclResult {};
        } catch (const std::exception& e) {
            return cuda::nccl::NcclResult {.code = ncclInternalError, .error_message = e.what()};
        }
    }

}  // namespace cuda::distributed
