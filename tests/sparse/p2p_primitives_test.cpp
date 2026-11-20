/**
 * @file p2p_primitives_test.cpp
 * @brief Contracts for the peer-to-peer send/receive primitives
 *
 * include/cuda/pipeline/p2p_primitives.h moves activations between pipeline
 * stages on different GPUs. It had no tests, and three defects lived in it:
 *
 * - The constructor called cudaEventCreate and cudaDeviceEnablePeerAccess
 *   without first selecting src_device_. Both act on the CURRENT device while
 *   the arguments name a specific pair, so enabling peer access for a pair the
 *   thread was not on failed with "invalid argument" — a buffer for devices 4
 *   and 5 could not be constructed at all unless something else had already
 *   selected device 4.
 *
 * - A second P2PSendRecv for the same pair threw "peer access is already
 *   enabled". The state it wanted is the state that already held.
 *
 * - send_async() returned silently when no destination had been registered, so
 *   the data was dropped and the receiver waited for bytes that were never
 *   queued. The op is also called in the documented order (send, then receive)
 *   in the header's own example, which made this reachable immediately.
 */

#include <cuda/pipeline/p2p_primitives.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace cuda::pipeline::test {

    namespace {

        int current_device() {
            int device = -1;
            cudaGetDevice(&device);
            return device;
        }

        // A device pair the rest of the suite does not touch, so enabling peer access
        // here cannot collide with another test's state.
        constexpr int kSrcDevice = 0;
        constexpr int kDstDevice = 1;

        bool peer_pair_available() {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
                return false;
            }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, kSrcDevice, kDstDevice) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return can_access != 0;
        }

    }  // namespace

    TEST(P2PSendRecvTest, ExposesTheDevicePairItWasBuiltFor) {
        // Constructing the pair needs both devices to exist, independently of whether
        // they can reach each other. The case was missing this guard while every other
        // case in the file has one, so running with a single visible device failed with
        // "invalid device ordinal" from the constructor instead of skipping:
        //
        //   CUDA_VISIBLE_DEVICES=5 ./bin/nova-tests
        //     --gtest_filter=P2PSendRecvTest.ExposesTheDevicePairItWasBuiltFor
        //   p2p_primitives.cpp:36 - CUDA error: invalid device ordinal
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "needs two visible devices, found " << device_count;
        }

        P2PSendRecv p2p(kSrcDevice, kDstDevice);

        EXPECT_EQ(p2p.src_device(), kSrcDevice);
        EXPECT_EQ(p2p.dst_device(), kDstDevice);

        // On this hardware the pair supports peer access; the flag is what the
        // operations gate on.
        EXPECT_EQ(p2p.has_peer_access(), peer_pair_available());
    }

    // Regression: enabling peer access happened on whatever device the thread was
    // on, so constructing a pair the thread was not sitting on failed.
    TEST(P2PSendRecvTest, ConstructsRegardlessOfTheCallersCurrentDevice) {
        if (!peer_pair_available()) {
            GTEST_SKIP() << "no peer access between the test devices";
        }

        const int original = current_device();
        // Sit on the destination, then build a pair whose source is the other one.
        ASSERT_EQ(cudaSetDevice(kDstDevice), cudaSuccess);

        EXPECT_NO_THROW({
            P2PSendRecv p2p(kSrcDevice, kDstDevice);
            EXPECT_EQ(p2p.src_device(), kSrcDevice);
        });

        EXPECT_EQ(current_device(), kDstDevice) << "constructing must not migrate the calling thread";
        EXPECT_EQ(cudaSetDevice(original), cudaSuccess);
    }

    // Regression: this threw "peer access is already enabled" once the first
    // object had turned it on, making the pair unusable for a second stage.
    TEST(P2PSendRecvTest, TwoObjectsForTheSamePairCanCoexist) {
        if (!peer_pair_available()) {
            GTEST_SKIP() << "no peer access between the test devices";
        }

        EXPECT_NO_THROW({
            P2PSendRecv first(kSrcDevice, kDstDevice);
            P2PSendRecv second(kSrcDevice, kDstDevice);
            EXPECT_TRUE(first.has_peer_access());
            EXPECT_TRUE(second.has_peer_access());
        });
    }

    TEST(P2PSendRecvTest, SendAndReceiveDoNotMigrateTheCaller) {
        if (!peer_pair_available()) {
            GTEST_SKIP() << "no peer access between the test devices";
        }

        const int original = current_device();
        const size_t bytes = 64 * sizeof(float);

        // cudaMemcpyPeerAsync requires device pointers on both sides, so allocate
        // one buffer per device rather than using host vectors.
        void* source = nullptr;
        void* destination = nullptr;
        ASSERT_EQ(cudaSetDevice(kSrcDevice), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&source, bytes), cudaSuccess);
        ASSERT_EQ(cudaSetDevice(kDstDevice), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&destination, bytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(destination, 0, bytes), cudaSuccess);

        P2PSendRecv p2p(kSrcDevice, kDstDevice);
        ASSERT_EQ(cudaSetDevice(kDstDevice), cudaSuccess);
        const int before = current_device();

        p2p.recv_async(destination, bytes, nullptr);
        EXPECT_EQ(current_device(), before) << "recv_async must not migrate the caller";

        p2p.send_async(source, bytes, nullptr);
        EXPECT_EQ(current_device(), before) << "send_async must not migrate the caller";

        p2p.wait_send();
        p2p.wait_recv();
        EXPECT_EQ(current_device(), before) << "waits must not migrate the caller";

        ASSERT_EQ(cudaSetDevice(kDstDevice), cudaSuccess);
        cudaFree(destination);
        ASSERT_EQ(cudaSetDevice(kSrcDevice), cudaSuccess);
        cudaFree(source);
        EXPECT_EQ(cudaSetDevice(original), cudaSuccess);
    }

    // Regression: this returned without doing anything, silently dropping the data.
    TEST(P2PSendRecvTest, SendWithoutARegisteredDestinationFailsLoudly) {
        if (!peer_pair_available()) {
            GTEST_SKIP() << "no peer access between the test devices";
        }

        P2PSendRecv p2p(kSrcDevice, kDstDevice);
        std::vector<float> source(32, 2.0f);

        EXPECT_THROW(p2p.send_async(source.data(), source.size() * sizeof(float), nullptr), std::runtime_error) << "a send with nowhere to go must not report success";
    }

    TEST(P2PSendRecvTest, OperationsWithoutPeerAccessFailLoudly) {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);

        // Pick a pair the topology says cannot reach each other, if any.
        int unsupported_src = -1, unsupported_dst = -1;
        for (int a = 0; a < count && unsupported_src < 0; ++a) {
            for (int b = 0; b < count; ++b) {
                if (a == b) {
                    continue;
                }
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess) {
                    (void)cudaGetLastError();
                    continue;
                }
                if (!can) {
                    unsupported_src = a;
                    unsupported_dst = b;
                    break;
                }
            }
        }
        if (unsupported_src < 0) {
            GTEST_SKIP() << "every device pair on this host supports peer access";
        }

        P2PSendRecv p2p(unsupported_src, unsupported_dst);
        ASSERT_FALSE(p2p.has_peer_access());

        std::vector<float> buffer(16, 0.0f);
        EXPECT_THROW(p2p.send_async(buffer.data(), sizeof(float) * 16, nullptr), std::runtime_error);
        EXPECT_THROW(p2p.recv_async(buffer.data(), sizeof(float) * 16, nullptr), std::runtime_error);
    }

    TEST(P2PCreateConnectionsTest, ConnectionsMatchTheMeshsPeerAccess) {
        // DeviceMesh is a process-wide singleton, so it cannot be constructed here;
        // initialise it and compare what create_p2p_connections builds against the
        // pair-access facts it reports.
        auto& mesh = ::cuda::mesh::DeviceMesh::instance();
        if (mesh.device_count() <= 0) {
            GTEST_SKIP() << "no CUDA devices to build a mesh from";
        }

        const auto connections = create_p2p_connections(mesh);

        const int count = mesh.device_count();
        int expected = 0;
        for (int i = 0; i < count; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (mesh.can_access_peer(i, j)) {
                    ++expected;
                }
            }
        }
        EXPECT_EQ(static_cast<int>(connections.size()), expected) << "one connection per ordered pair that supports peer access";

        for (const auto& connection : connections) {
            ASSERT_NE(connection, nullptr);
            EXPECT_LT(connection->src_device(), connection->dst_device()) << "create_p2p_connections only builds the i < j direction";
        }
    }

}  // namespace cuda::pipeline::test
