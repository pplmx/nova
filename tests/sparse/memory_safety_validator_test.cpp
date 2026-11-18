/**
 * @file memory_safety_validator_test.cpp
 * @brief MemoryPoisonGuard touched device memory from the host, and two checks could not fail
 *
 * Three defects in one small header:
 *
 *   MemoryPoisonGuard::is_poisoned()  read the pointee byte by byte with a host loop, and the
 *                                     destructor memset it the same way. The pointee is device
 *                                     memory — check_uninitialized in the same file stages a
 *                                     host copy and says why in a comment. Measured before the
 *                                     fix: constructing a guard over a cudaMalloc'd block and
 *                                     letting it run segfaulted, exit code 139, in both
 *                                     is_poisoned() and the destructor.
 *
 *   check_double_free(ptr)            `return true;` unconditionally. It reported nothing.
 *
 *   validate()                        set valid = true and returned. A validation that cannot
 *                                     fail is indistinguishable from no validation.
 *
 * Nothing in the tree called any of these, so no existing test noticed.
 */

#include <cuda/testing/memory_safety.h>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <string>

namespace memory_safety_validator_test {

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

class MemorySafetyValidatorTest : public ::testing::Test {
protected:
    void TearDown() override {
        cuda::testing::MemorySafetyValidator::instance().reset();
        (void)cudaGetLastError();
    }
};

TEST_F(MemorySafetyValidatorTest, PoisonGuardOnDeviceMemoryDoesNotFault) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 256), cudaSuccess);

    {
        // Both the scan and the destructor used to dereference device_ptr from the host.
        cuda::testing::MemoryPoisonGuard guard(device_ptr, 256);
        EXPECT_FALSE(guard.is_poisoned()) << "a fresh allocation holds no poison byte";
    }

    EXPECT_EQ(cudaFree(device_ptr), cudaSuccess) << "the context must still be usable";
}

TEST_F(MemorySafetyValidatorTest, TheDestructorLeavesPoisonBehindWhenNotMarkedValid) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 64), cudaSuccess);
    ASSERT_EQ(cudaMemset(device_ptr, 0, 64), cudaSuccess);

    {
        cuda::testing::MemoryPoisonGuard guard(device_ptr, 64);
        // Not marked valid, so the destructor poisons it.
    }

    // Confirm through a host copy rather than by reading the device pointer directly, which is
    // the mistake this file is about.
    unsigned char host[64];
    ASSERT_EQ(cudaMemcpy(host, device_ptr, 64, cudaMemcpyDeviceToHost), cudaSuccess);
    bool all_poison = true;
    for (unsigned char byte : host) {
        if (byte != 0xFE) {
            all_poison = false;
            break;
        }
    }
    EXPECT_TRUE(all_poison) << "an unmarked region is poisoned on destruction";

    EXPECT_EQ(cudaFree(device_ptr), cudaSuccess);
}

TEST_F(MemorySafetyValidatorTest, MarkValidKeepsTheDataIntact) {
    if (!device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 32), cudaSuccess);
    ASSERT_EQ(cudaMemset(device_ptr, 0x11, 32), cudaSuccess);

    {
        cuda::testing::MemoryPoisonGuard guard(device_ptr, 32);
        guard.mark_valid();
    }

    unsigned char host[32];
    ASSERT_EQ(cudaMemcpy(host, device_ptr, 32, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host[0], 0x11) << "mark_valid must suppress the poison write";

    EXPECT_EQ(cudaFree(device_ptr), cudaSuccess);
}

TEST_F(MemorySafetyValidatorTest, CheckDoubleFreeReportsARepeatedAddress) {
    auto& validator = cuda::testing::MemorySafetyValidator::instance();
    validator.reset();
    validator.enable();

    int probe = 0;
    EXPECT_TRUE(validator.check_double_free(&probe)) << "the first sighting is accepted";
    EXPECT_FALSE(validator.check_double_free(&probe))
        << "the same address a second time is what this is meant to catch";
    EXPECT_GT(validator.error_count(), 0u) << "the rejection is counted";

    int other = 0;
    EXPECT_TRUE(validator.check_double_free(&other)) << "a different address is fine";
}

TEST_F(MemorySafetyValidatorTest, CheckDoubleFreeRejectsNull) {
    auto& validator = cuda::testing::MemorySafetyValidator::instance();
    validator.reset();
    validator.enable();
    EXPECT_FALSE(validator.check_double_free(nullptr));
}

TEST_F(MemorySafetyValidatorTest, ValidateReportsTheRecordedErrors) {
    auto& validator = cuda::testing::MemorySafetyValidator::instance();
    validator.reset();
    validator.enable();

    auto clean = validator.validate();
    EXPECT_TRUE(clean.valid) << "nothing has gone wrong yet";

    // Provoke one recorded error.
    EXPECT_FALSE(validator.validate_allocation(nullptr, 16));

    auto after = validator.validate();
    EXPECT_FALSE(after.valid) << "a recorded error must make the summary invalid";
    EXPECT_FALSE(after.error_message.empty()) << "and it must say something about it";
    EXPECT_NE(after.error_message.find("error"), std::string::npos);
}

TEST_F(MemorySafetyValidatorTest, ADisabledValidatorSkipsAndCountsNothing) {
    auto& validator = cuda::testing::MemorySafetyValidator::instance();
    validator.reset();
    validator.disable();

    // A disabled validator short-circuits every check to true and records nothing, which is
    // what enable()/disable() are for. Pinned so a future change to the early returns shows up.
    EXPECT_TRUE(validator.validate_allocation(nullptr, 16));
    EXPECT_EQ(validator.error_count(), 0u);
    EXPECT_EQ(validator.validation_count(), 0u);
    EXPECT_TRUE(validator.validate().valid);

    validator.enable();
}

}  // namespace memory_safety_validator_test
