#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gtest/gtest.h>
#include <vector>

#include "cuda/production/graph_executor.h"
#include "cuda/production/memory_node.h"

namespace {

void reset() {
    cudaDeviceReset();
}

}

class GraphExecutorTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(GraphExecutorTest, DefaultConstruction) {
    cuda::production::GraphExecutor executor;
    EXPECT_FALSE(executor.is_capturing());
    EXPECT_FALSE(executor.is_instantiated());
    EXPECT_EQ(executor.param_count(), 0u);
}

TEST_F(GraphExecutorTest, CopyConstructionDeleted) {
    cuda::production::GraphExecutor executor;
    EXPECT_FALSE(std::is_copy_constructible_v<cuda::production::GraphExecutor>);
}

TEST_F(GraphExecutorTest, CopyAssignmentDeleted) {
    cuda::production::GraphExecutor executor;
    EXPECT_FALSE(std::is_copy_assignable_v<cuda::production::GraphExecutor>);
}

TEST_F(GraphExecutorTest, MoveConstruction) {
    cuda::production::GraphExecutor executor1;
    executor1.begin_capture();

    cuda::production::GraphExecutor executor2(std::move(executor1));
    EXPECT_FALSE(executor2.is_instantiated());
}

TEST_F(GraphExecutorTest, MoveAssignment) {
    cuda::production::GraphExecutor executor1;
    cuda::production::GraphExecutor executor2;

    executor1.begin_capture();
    executor1 = std::move(executor2);

    EXPECT_FALSE(executor1.is_capturing());
}

TEST_F(GraphExecutorTest, BeginEndCapture) {
    cuda::production::GraphExecutor executor;
    cuda::stream::Stream stream;

    EXPECT_FALSE(executor.is_capturing());

    executor.begin_capture(stream);
    EXPECT_TRUE(executor.is_capturing());

    auto graph = executor.end_capture();
    EXPECT_NE(graph, nullptr);
    EXPECT_FALSE(executor.is_capturing());
}

TEST_F(GraphExecutorTest, CaptureAndLaunch) {
    cuda::production::GraphExecutor executor;
    cuda::stream::Stream stream;

    // Memory allocation is NOT capturable: cudaMalloc inside cudaStreamBeginCapture
    // records "operation not permitted during stream capture" and corrupts the
    // capture, making end_capture() fail. Allocate before capture, free after.
    int* d_data = nullptr;
    cudaMalloc(&d_data, sizeof(int));

    executor.begin_capture(stream);

    // Use an async (capturable) op on the capture stream. Blocking cudaMemset
    // issues to the default stream and is neither captured nor allowed to
    // synchronize during a thread-local capture of another stream.
    cudaMemsetAsync(d_data, 0, sizeof(int), stream.get());

    executor.end_capture();
    executor.instantiate();

    EXPECT_TRUE(executor.is_instantiated());

    executor.launch(stream);
    stream.synchronize();

    cudaFree(d_data);
}

TEST_F(GraphExecutorTest, ScopedCaptureRAII) {
    cuda::production::GraphExecutor executor;
    cuda::stream::Stream stream;

    // Same as CaptureAndLaunch: cudaMalloc/cudaFree are not permitted inside an
    // active graph capture, so keep allocation outside the captured work.
    int* d_data = nullptr;
    cudaMalloc(&d_data, sizeof(int));

    {
        cuda::production::ScopedCapture capture(executor, stream.get());
        // Capturable async op on the capture stream (see CaptureAndLaunch).
        cudaMemsetAsync(d_data, 0, sizeof(int), stream.get());
    }

    EXPECT_TRUE(executor.is_instantiated());
    cudaFree(d_data);
}

TEST_F(GraphExecutorTest, UpdateParam) {
    cuda::production::GraphExecutor executor;

    int value = 42;
    executor.update_param(0, &value, sizeof(int));

    EXPECT_EQ(executor.param_count(), 1u);

    int new_value = 100;
    executor.update_param(0, &new_value, sizeof(int));

    EXPECT_EQ(executor.param_count(), 1u);
}

TEST_F(GraphExecutorTest, Reset) {
    cuda::production::GraphExecutor executor;
    cuda::stream::Stream stream;

    executor.begin_capture(stream);
    executor.end_capture();
    executor.instantiate();

    EXPECT_TRUE(executor.is_instantiated());

    executor.reset();

    EXPECT_FALSE(executor.is_capturing());
    EXPECT_FALSE(executor.is_instantiated());
    EXPECT_EQ(executor.param_count(), 0u);
}

TEST_F(GraphExecutorTest, ReuseAfterReset) {
    cuda::production::GraphExecutor executor;
    cuda::stream::Stream stream;

    executor.begin_capture(stream);
    executor.end_capture();
    executor.instantiate();
    executor.reset();

    executor.begin_capture(stream);
    auto graph = executor.end_capture();
    executor.instantiate();

    EXPECT_TRUE(executor.is_instantiated());
}

class MemoryNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};

TEST_F(MemoryNodeTest, DefaultConstruction) {
    cuda::production::MemoryNode node;
    EXPECT_EQ(node.get(), nullptr);
}

TEST_F(MemoryNodeTest, DeviceAllocation) {
    cuda::production::GraphExecutor executor;
    cuda::production::GraphMemoryManager manager;

    executor.begin_capture();
    auto node = manager.add_device_allocation(executor, executor.end_capture(), 1024);

    EXPECT_EQ(node.type(), cuda::production::MemoryType::Device);
    EXPECT_NE(node.ptr(), nullptr);
    EXPECT_EQ(node.size(), 1024u);
    EXPECT_EQ(manager.allocation_count(), 1u);

    manager.cleanup();
}

TEST_F(MemoryNodeTest, HostAllocation) {
    cuda::production::GraphExecutor executor;
    cuda::production::GraphMemoryManager manager;

    executor.begin_capture();
    auto node = manager.add_host_allocation(executor, executor.end_capture(), 2048);

    EXPECT_EQ(node.type(), cuda::production::MemoryType::HostPinned);
    EXPECT_NE(node.ptr(), nullptr);
    EXPECT_EQ(node.size(), 2048u);

    manager.cleanup();
}

TEST_F(MemoryNodeTest, ManagedAllocation) {
    cuda::production::GraphExecutor executor;
    cuda::production::GraphMemoryManager manager;

    executor.begin_capture();
    auto node = manager.add_managed_allocation(executor, executor.end_capture(), 4096);

    EXPECT_EQ(node.type(), cuda::production::MemoryType::Managed);
    EXPECT_NE(node.ptr(), nullptr);
    EXPECT_EQ(node.size(), 4096u);

    manager.cleanup();
}

TEST_F(MemoryNodeTest, TotalAllocatedTracking) {
    cuda::production::GraphExecutor executor;
    cuda::production::GraphMemoryManager manager;

    executor.begin_capture();
    // End the capture once and add both allocations to the same graph. Calling
    // end_capture() a second time throws (the executor is no longer capturing).
    cudaGraph_t graph = executor.end_capture();
    manager.add_device_allocation(executor, graph, 1024);
    manager.add_device_allocation(executor, graph, 2048);

    EXPECT_GE(manager.total_allocated(), 1024u + 2048u);

    manager.cleanup();
}

TEST_F(MemoryNodeTest, Cleanup) {
    cuda::production::GraphExecutor executor;
    cuda::production::GraphMemoryManager manager;

    {
        executor.begin_capture();
        manager.add_device_allocation(executor, executor.end_capture(), 1024);
    }

    EXPECT_EQ(manager.allocation_count(), 1u);
    manager.cleanup();
    EXPECT_EQ(manager.allocation_count(), 0u);
}

TEST_F(MemoryNodeTest, ScopedGraphBuffer) {
    cuda::production::GraphExecutor executor;
    cuda::production::GraphMemoryManager manager;
    cuda::stream::Stream stream;

    executor.begin_capture(stream);

    cuda::production::ScopedGraphBuffer<float> buffer(manager, executor, nullptr, 100);

    EXPECT_NE(buffer.get(), nullptr);
    EXPECT_EQ(buffer.size(), 100u);

    executor.end_capture();

    cudaFree(buffer.get());
}

TEST_F(MemoryNodeTest, ScopedGraphBufferMove) {
    cuda::production::GraphExecutor executor;
    cuda::production::GraphMemoryManager manager;

    executor.begin_capture();
    cuda::production::ScopedGraphBuffer<float> buffer1(manager, executor, nullptr, 50);

    cuda::production::ScopedGraphBuffer<float> buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.size(), 50u);
    EXPECT_EQ(buffer1.get(), nullptr);

    executor.end_capture();

    cudaFree(buffer2.get());
}
