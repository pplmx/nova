/**
 * @file upload_empty_graph_test.cpp
 * @brief CSRGraph::upload must reject a graph whose host arrays were never allocated
 *
 * The default constructor leaves row_offsets, columns and weights null. upload() copies
 *
 *     cudaMemcpy(d_row_offsets, row_offsets, (num_vertices + 1) * sizeof(int), ...)
 *
 * so for a default-constructed graph it asks the driver to read one int from a null host
 * pointer. num_vertices is zero, so the length is not zero — the copy is attempted.
 *
 * The existing DefaultConstruction case checks the counts only and never uploads.
 */

#include <cuda/graph/csr_graph.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace upload_empty_graph_test {

    class UploadEmptyGraphTest : public ::testing::Test {
    protected:
        void TearDown() override { (void)cudaGetLastError(); }
    };

    TEST_F(UploadEmptyGraphTest, UploadingADefaultConstructedGraphFailsLoudly) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "no CUDA device";
        }

        cuda::graph::CSRGraph graph;  // every array is null

        // The copy would read one int through a null host pointer. Before the guard this threw
        // only because the driver rejected the source; the check below asserts the named contract
        // instead, so the case fails if the guard is removed rather than passing on driver luck.
        auto call = [&] { graph.upload(); };

        bool named_the_precondition = false;
        try {
            call();
        } catch (const std::invalid_argument& e) {
            named_the_precondition = std::string(e.what()).find("row_offsets is null") != std::string::npos;
        } catch (...) {
            // A driver-level failure means the guard did not fire. Fall through to the assertion.
        }
        EXPECT_TRUE(named_the_precondition) << "upload() must reject a graph with no host arrays by name";
        (void)cudaGetLastError();
    }

}  // namespace upload_empty_graph_test
