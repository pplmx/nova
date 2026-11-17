/**
 * @file csr_requires_upload_test.cpp
 * @brief The device-side CSR operations require upload() and never said so
 *
 * compute_degrees(), csr_mv() and friends read graph.d_row_offsets / graph.d_columns, which
 * CSRGraph leaves null until upload() is called. None of the declarations in csr_graph.h
 * documents that requirement, and none of the implementations checks it: a caller that forgets
 * gets a kernel launch reading through null device pointers.
 *
 * Both facts are exercised here. The point is not that a missing upload must throw — the
 * current design lets the driver report it asynchronously — but that the requirement is
 * invisible from the header, so the test records what actually happens.
 */

#include <cuda/graph/csr_graph.h>
#include <cuda/memory/buffer.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace csr_requires_upload_test {

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

class CsrRequiresUploadTest : public ::testing::Test {
protected:
    void TearDown() override { (void)cudaGetLastError(); }
};

TEST_F(CsrRequiresUploadTest, AGraphThatWasNeverUploadedHoldsNoDeviceArrays) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::graph::CSRGraph graph(3, 2);
    // upload() was never called. This is the state in which compute_degrees(), csr_mv() and
    // friends would pass null device pointers to their kernels.
    EXPECT_EQ(graph.d_row_offsets, nullptr)
        << "the device arrays stay null until upload() is called";
    EXPECT_EQ(graph.d_columns, nullptr);

    // Not calling those functions here: a launch reading through a null device pointer does not
    // fail an assertion, it corrupts the CUDA context. An earlier version of this file did
    // exactly that, and every later case in the process then failed.
    graph.upload();
    EXPECT_NE(graph.d_row_offsets, nullptr) << "upload() is what allocates and fills them";
}

TEST_F(CsrRequiresUploadTest, AnUploadedGraphComputesDegreesNormally) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    cuda::graph::CSRGraph graph(3, 2);
    graph.row_offsets[0] = 0;
    graph.row_offsets[1] = 1;
    graph.row_offsets[2] = 2;
    graph.row_offsets[3] = 2;
    graph.columns[0] = 1;
    graph.columns[1] = 2;
    graph.upload();

    cuda::memory::Buffer<int> degrees(3);
    ASSERT_NO_THROW(cuda::graph::compute_degrees(graph, degrees.data()));

    std::vector<int> host(3, -1);
    degrees.copy_to(host.data(), 3);
    EXPECT_EQ(host[0], 1) << "vertex 0 has one out-edge";
    EXPECT_EQ(host[1], 1) << "vertex 1 has one out-edge";
    EXPECT_EQ(host[2], 0) << "vertex 2 has none";
}

}  // namespace csr_requires_upload_test
