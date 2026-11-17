/**
 * @file documented_examples_test.cpp
 * @brief The usage examples in the headers must compile and work
 *
 * A header's @code block is what a caller copies. Nothing compiles those blocks, so they rot
 * silently when a signature changes — and unlike a test, rot there reaches users before it
 * reaches CI.
 *
 * There are 73 @code blocks in include/. This file covers the ones that name a concrete call
 * with arguments; the rest are sketches without a call. The DistributedMatmulOptions example
 * and the all_gather and DeviceMesh calls below were checked against the declarations while
 * writing this, and the single-GPU fallback the distributed headers promise was confirmed in
 * the implementations (device_count() <= 1 branches that copy the input through).
 *
 * Compiling this file is most of the check. The cases then exercise what the examples claim
 * so that a block can still be wrong in a way that builds.
 */

#include <cuda/distributed/all_gather.h>
#include <cuda/distributed/matmul.h>
#include <cuda/distributed/reduce.h>
#include <cuda/mesh/device_mesh.h>
#include <cuda/nccl/nccl_context.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <numeric>
#include <vector>

namespace documented_examples_test {

namespace {

bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count > 0;
}

}  // namespace

// matmul.h documents:
//   DistributedMatmulOptions opts; opts.alpha = 1.0f; opts.beta = 0.0f;
//   DistributedMatmul::matmul(A, B, C, m, n, k, opts);
TEST(DocumentedExamplesTest, TheDistributedMatmulOptionsExampleBuildsAndComputes) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::distributed::DistributedMatmulOptions opts;
    opts.alpha = 1.0f;
    opts.beta = 0.0f;

    // The example takes device pointers; the headers now say so. My first version of this
    // case passed host vectors and got "an illegal memory access was encountered", which is
    // exactly what the documented code would have done for a reader who copied it.
    const std::vector<float> a{1, 2,
                               3, 4};
    const std::vector<float> b{5, 6,
                               7, 8};
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    ASSERT_EQ(cudaMalloc(&d_a, a.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, b.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, 4 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);

    cuda::distributed::DistributedMatmul::matmul(d_a, d_b, d_c, 2, 2, 2, opts);

    std::vector<float> c(4, 0.0F);
    ASSERT_EQ(cudaMemcpy(c.data(), d_c, 4 * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    // [1 2] . [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    EXPECT_NEAR(c[0], 19.0F, 1e-4F);
    EXPECT_NEAR(c[1], 22.0F, 1e-4F);
    EXPECT_NEAR(c[2], 43.0F, 1e-4F);
    EXPECT_NEAR(c[3], 50.0F, 1e-4F);

    (void)cudaFree(d_a);
    (void)cudaFree(d_b);
    (void)cudaFree(d_c);
}

// all_gather.h documents:
//   DistributedAllGather::all_gather(send_data, recv_data, count);
// with "Single-GPU fallback: Copy send_data to recv_data (same size)".
TEST(DocumentedExamplesTest, TheAllGatherExampleFollowsItsSingleGpuFallback) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    // initialize() first: device_count() is 1 before the mesh has enumerated, so asking
    // without it can send this case down the single-GPU branch on a four-device machine. That
    // ordering was the reason it passed alone and failed under ctest.
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();
    const int device_count = mesh.device_count();
    if (device_count > 1) {
        // With more than one device the call needs an initialised NCCL context, which this
        // test does not set up. That path reports exactly that, naming what to do:
        //
        //   "multi-GPU all-gather requires an initialized NCCL context (call
        //    NcclContext::instance().initialize() from one thread per rank); the single-GPU
        //    fallback applies only when device_count <= 1"
        //
        // Asserting that message keeps the multi-device path covered without a cluster.
        // Note on shape: the NcclException carries __FILE__ and __LINE__, and gtest reports a
        // thrown exception from EXPECT_THROW as an uncaught failure at "unknown file" when
        // the type does not match exactly. Catching it directly and asserting on the message
        // keeps the failure readable and does not depend on the macro's comma handling.
        std::vector<float> send_data{1.0F, 2.0F, 3.0F, 4.0F};
        bool threw = false;
        std::string message;
        try {
            cuda::distributed::DistributedAllGather::all_gather(send_data.data(),
                                                                send_data.data(),
                                                                send_data.size());
        } catch (const std::runtime_error& error) {
            threw = true;
            message = error.what();
        }
        EXPECT_TRUE(threw) << "the multi-device path should refuse without an NCCL context";
        EXPECT_NE(message.find("NCCL"), std::string::npos)
            << "the message should name NCCL as the missing piece: " << message;
        return;
    }

    const std::vector<float> send_data{1.0F, 2.0F, 3.0F, 4.0F};
    float* d_send = nullptr;
    float* d_recv = nullptr;
    ASSERT_EQ(cudaMalloc(&d_send, send_data.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_recv, send_data.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_recv, 0, send_data.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_send, send_data.data(), send_data.size() * sizeof(float),
                         cudaMemcpyHostToDevice), cudaSuccess);

    cuda::distributed::DistributedAllGather::all_gather(d_send, d_recv, send_data.size());

    std::vector<float> recv_data(send_data.size(), -1.0F);
    ASSERT_EQ(cudaMemcpy(recv_data.data(), d_recv, send_data.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    // The header says the input is copied through when there is only one device.
    for (size_t i = 0; i < send_data.size(); ++i) {
        EXPECT_FLOAT_EQ(recv_data[i], send_data[i]) << "index " << i;
    }

    (void)cudaFree(d_send);
    (void)cudaFree(d_recv);
}

// device_mesh.h documents:
//   auto& mesh = cuda::mesh::DeviceMesh::instance(); mesh.initialize();
TEST(DocumentedExamplesTest, TheDeviceMeshExampleIsCallableTwice) {
    auto& mesh = cuda::mesh::DeviceMesh::instance();
    mesh.initialize();
    const int first = mesh.device_count();

    // initialize() is documented as idempotent by the CAS in its implementation; calling it
    // again must not change what it reported.
    mesh.initialize();
    EXPECT_EQ(mesh.device_count(), first);
    EXPECT_GE(first, 1) << "the mesh reports at least one device by design";
}

}  // namespace documented_examples_test
