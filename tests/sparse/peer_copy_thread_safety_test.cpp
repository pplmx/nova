/**
 * @file peer_copy_thread_safety_test.cpp
 * @brief PeerCopy's enabled-pairs set has no lock, against what its header claimed
 *
 * include/cuda/mesh/peer_copy.h said "Provides thread-safe async P2P copy between GPUs".
 * enable_peer_access() reads and writes enabled_peer_pairs_, a std::set, with no mutex, and it
 * is a public method — the header's own example calls it directly. Two threads enabling peer
 * access at the same time race on that set. copy_async() never touches it, so concurrent
 * copies are fine; the constraint is on enabling.
 *
 * The header now says this. These cases pin the single-threaded contract that remains, since
 * a data race that happens to pass proves nothing about the race.
 */

#include <cuda/mesh/peer_copy.h>
#include <cuda/device/error.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace peer_copy_thread_safety_test {

namespace {

bool two_devices_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return count >= 2;
}

}  // namespace

TEST(PeerCopyThreadSafetyTest, EnablingAPairThatIsUnavailableFailsLoudly) {
    if (!two_devices_available()) {
        GTEST_SKIP() << "need at least two devices";
    }

    cuda::mesh::PeerCopy copier;

    // The set that enable_peer_access() maintains is keyed on the ordered device pair, and
    // the two calls below exercise it without needing P2P to exist on this machine. The
    // existing PeerCopyTest.PeerAccessAvailableMultiGPU notes that the answer "may be true or
    // false depending on hardware", so requiring true here would make this case a hardware
    // probe rather than a check of the class.
    const bool available = copier.peer_access_available(0, 1);
    if (!available) {
        // A pair that cannot access itself fails loudly instead of silently doing nothing —
        // the behaviour the set's lookup guards.
        EXPECT_THROW(copier.enable_peer_access(0, 1), cuda::device::CudaException);
        return;
    }

    EXPECT_NO_THROW(copier.enable_peer_access(0, 1));
}

TEST(PeerCopyThreadSafetyTest, SelfAccessNeedsNoEnablingAndNeverTouchesTheSet) {
    if (!two_devices_available()) {
        GTEST_SKIP() << "need at least two devices";
    }

    cuda::mesh::PeerCopy copier;

    // Self-access returns before the set is consulted (enable_peer_access's first branch), so
    // it is safe no matter what state the set is in — the one path that is genuinely
    // thread-safe today.
    EXPECT_TRUE(copier.peer_access_available(0, 0));
    EXPECT_NO_THROW(copier.enable_peer_access(0, 0));
    EXPECT_NO_THROW(copier.enable_peer_access(1, 1));
}

}  // namespace peer_copy_thread_safety_test
