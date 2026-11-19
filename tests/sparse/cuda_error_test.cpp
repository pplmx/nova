/**
 * @file cuda_error_test.cpp
 * @brief Contracts for the structured CUDA error layer
 *
 * include/cuda/error/cuda_error.hpp is 300 lines of documented API — error
 * info, an std::error_category, a throwing RAII guard and two macros — with no
 * test coverage at all. The recovery hints were written against numeric
 * literals rather than the cudaError_t enumerators, and several of them had
 * drifted:
 *
 *   - cudaErrorNotReady is 600, but the hint was attached to 500, so the most
 *     common "not done yet" error fell through to the generic message;
 *   - 201 is cudaErrorDeviceUninitialized, not InvalidContext, so a device
 *     context problem was answered with "call cudaSetDevice()";
 *   - 3 (the commented cudaErrorNotInitialized) does not exist in modern CUDA.
 *
 * These tests assert against the enumerators, so the mapping cannot drift back.
 */

#include <cuda/error/cublas_error.hpp>
#include <cuda/error/cuda_error.hpp>

#include <gtest/gtest.h>

#include <string>
#include <system_error>

namespace nova::error::test {

    namespace {

        constexpr const char* kGenericHint = "Consult CUDA documentation for this error code";

        std::string_view hint_for(cudaError_t err) {
            return static_cast<const cuda_error_category&>(cuda_category()).recovery_hint(static_cast<int>(err));
        }

    }  // namespace

    TEST(CudaErrorTest, CategoryHasAStableName) {
        EXPECT_STREQ(cuda_category().name(), "cuda");
    }

    TEST(CudaErrorTest, MessageMatchesTheRuntimeString) {
        const auto& cat = cuda_category();
        EXPECT_EQ(cat.message(static_cast<int>(cudaSuccess)), cudaGetErrorString(cudaSuccess));
        EXPECT_EQ(cat.message(static_cast<int>(cudaErrorMemoryAllocation)), cudaGetErrorString(cudaErrorMemoryAllocation));
    }

    // Each of these errors has a specific, actionable hint. Asserting them by
    // enumerator rather than by number is the point: the old code keyed off
    // literals and silently attached hints to the wrong errors.
    TEST(CudaErrorTest, ActionableErrorsHaveSpecificHints) {
        const cudaError_t actionable[] = {
            cudaErrorInvalidValue,
            cudaErrorMemoryAllocation,
            cudaErrorNoDevice,
            cudaErrorInvalidDevice,
            cudaErrorNotReady,
            cudaErrorIllegalAddress,
            cudaErrorFileNotFound,
            cudaErrorDeviceUninitialized,
        };

        for (const cudaError_t err : actionable) {
            const std::string_view hint = hint_for(err);
            EXPECT_FALSE(hint.empty()) << "no hint for " << static_cast<int>(err);
            EXPECT_NE(hint, kGenericHint) << "error " << static_cast<int>(err) << " (" << cudaGetErrorString(err) << ") fell through to the generic hint";
        }
    }

    TEST(CudaErrorTest, CudaErrorNotReadyKeepsItsHint) {
        // 600, not the 500 the literal keyed on: an async operation that has not
        // finished is the single most common non-fatal error and must not lose its
        // guidance.
        EXPECT_EQ(static_cast<int>(cudaErrorNotReady), 600);
        EXPECT_NE(hint_for(cudaErrorNotReady), kGenericHint);
    }

    TEST(CudaErrorTest, UninitializedDeviceGetsItsOwnHint) {
        // 201 is cudaErrorDeviceUninitialized. The old table mapped 201 to
        // "Invalid device context - call cudaSetDevice()", which reports a
        // different problem than the one that occurred.
        EXPECT_EQ(static_cast<int>(cudaErrorDeviceUninitialized), 201);

        const std::string_view hint = hint_for(cudaErrorDeviceUninitialized);
        EXPECT_NE(hint, kGenericHint);
        EXPECT_EQ(hint.find("Invalid device context"), std::string_view::npos) << "an uninitialized device is not an invalid context: " << hint;
        // It must talk about initialization, which is the actual failure.
        EXPECT_NE(hint.find("initializ"), std::string_view::npos) << "the hint should mention initialization: " << hint;
    }

    TEST(CudaErrorTest, UnmappedErrorFallsBackToTheGenericHint) {
        // An error with no specific guidance still produces a usable string.
        const std::string_view hint = hint_for(static_cast<cudaError_t>(99999));
        EXPECT_EQ(hint, kGenericHint);
    }

    TEST(CudaErrorTest, ErrorInfoFormatsOperationFileLineAndDevice) {
        cuda_error_info info {};
        info.code = cudaErrorMemoryAllocation;
        info.operation = "cudaMalloc";
        info.file = "memory.cu";
        info.line = 42;
        info.device_id = 0;

        const std::string message = info.message();
        EXPECT_NE(message.find("cudaMalloc"), std::string::npos);
        EXPECT_NE(message.find(cudaGetErrorString(cudaErrorMemoryAllocation)), std::string::npos);
        EXPECT_NE(message.find("memory.cu:42"), std::string::npos);
        EXPECT_NE(message.find("(device 0)"), std::string::npos);
    }

    TEST(CudaErrorTest, ErrorInfoOmitsUnsetContext) {
        cuda_error_info info {};
        info.code = cudaErrorInvalidValue;

        const std::string message = info.message();
        EXPECT_EQ(message.find("failed:"), std::string::npos) << "no operation was set";
        EXPECT_EQ(message.find(" at "), std::string::npos) << "no file/line was set";
        EXPECT_EQ(message.find("device "), std::string::npos) << "no device was set";
    }

    TEST(CudaErrorTest, ErrorInfoHintAgreesWithTheCategory) {
        cuda_error_info info {};
        info.code = cudaErrorIllegalAddress;

        EXPECT_EQ(info.recovery_hint(), hint_for(cudaErrorIllegalAddress));
    }

    TEST(CudaErrorTest, ExceptionCarriesTheErrorAndContext) {
        cuda_exception ex(cudaErrorInvalidValue, "cudaSetDevice", "device.cu", 7, 2);

        EXPECT_EQ(ex.info().code, cudaErrorInvalidValue);
        EXPECT_STREQ(ex.info().operation, "cudaSetDevice");
        EXPECT_STREQ(ex.info().file, "device.cu");
        EXPECT_EQ(ex.info().line, 7);
        EXPECT_EQ(ex.info().device_id, 2);

        // It is a std::system_error, so the generic error_code machinery works.
        EXPECT_EQ(ex.code().category(), cuda_category());
        EXPECT_EQ(ex.code().value(), static_cast<int>(cudaErrorInvalidValue));
    }

    TEST(CudaErrorTest, GuardPassesSuccessAndReportsOk) {
        cuda_error_guard guard("test", 0);
        EXPECT_TRUE(guard.ok());
        EXPECT_NO_THROW(guard.check(cudaSuccess));
        EXPECT_TRUE(guard.ok());
    }

    TEST(CudaErrorTest, GuardThrowsAndRecordsTheFirstFailure) {
        cuda_error_guard guard("cudaMalloc", 0, nullptr, __FILE__, __LINE__);

        bool threw = false;
        try {
            guard.check(cudaErrorMemoryAllocation);
        } catch (const cuda_exception& e) {
            threw = true;
            EXPECT_EQ(e.info().code, cudaErrorMemoryAllocation);
            EXPECT_STREQ(e.info().operation, "cudaMalloc");
        }
        EXPECT_TRUE(threw);
        EXPECT_FALSE(guard.ok());
        EXPECT_EQ(guard.info().code, cudaErrorMemoryAllocation);
    }

    // A real failing CUDA call through the macro, not just the guard in isolation.
    // Draining sticky error state afterwards keeps the failure from leaking into
    // later tests.
    TEST(CudaErrorTest, CheckMacroThrowsOnAGenuineFailure) {
        // The macro expands to a guard plus check(); wrap it in a lambda because
        // the statement contains commas that the EXPECT_THROW macro would split on.
        auto invoke = [] { NOVA_CHECK(static_cast<cudaError_t>(cudaErrorInvalidValue)); };
        EXPECT_THROW(invoke(), cuda_exception);
        [[maybe_unused]] cudaError_t drained = cudaGetLastError();
    }

    TEST(CudaErrorTest, AutoDetectedDeviceIsNotNegative) {
        int expected_device = -1;
        if (cudaGetDevice(&expected_device) != cudaSuccess) {
            GTEST_SKIP() << "no CUDA device available";
        }
        cuda_error_guard guard("auto", -1);
        EXPECT_EQ(guard.info().device_id, expected_device);
    }

    // ---------------------------------------------------------------- cuBLAS ----
    //
    // The same defect lived in the cuBLAS table: it was keyed on literals that did
    // not match the enumerators (11 is MAPPING_ERROR, not INTERNAL_ERROR; 13 is
    // EXECUTION_FAILED, not INVALID_HANDLE; 14 is INTERNAL_ERROR, not
    // INVALID_POINTER), and two of the statuses it named do not exist in the
    // modern API.

    TEST(CublasErrorTest, EveryNamedStatusHasItsOwnHint) {
        const cublasStatus_t statuses[] = {
            CUBLAS_STATUS_NOT_INITIALIZED,
            CUBLAS_STATUS_ALLOC_FAILED,
            CUBLAS_STATUS_INVALID_VALUE,
            CUBLAS_STATUS_ARCH_MISMATCH,
            CUBLAS_STATUS_MAPPING_ERROR,
            CUBLAS_STATUS_EXECUTION_FAILED,
            CUBLAS_STATUS_INTERNAL_ERROR,
            CUBLAS_STATUS_NOT_SUPPORTED,
            CUBLAS_STATUS_LICENSE_ERROR,
        };

        constexpr const char* kGeneric = "Consult cuBLAS documentation for this status code";
        for (const cublasStatus_t status : statuses) {
            const std::string_view hint = static_cast<const cublas_error_category&>(cublas_category()).recovery_hint(static_cast<int>(status));
            EXPECT_FALSE(hint.empty());
            EXPECT_NE(hint, kGeneric) << "status " << cublas_status_name(status) << " fell through";
        }
    }

    TEST(CublasErrorTest, HintsAreNotAttachedToTheWrongStatus) {
        const auto& cat = static_cast<const cublas_error_category&>(cublas_category());

        // 13 is EXECUTION_FAILED. The old table answered it with handle advice.
        EXPECT_EQ(static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED), 13);
        EXPECT_EQ(cat.recovery_hint(13).find("handle"), std::string_view::npos) << "execution failure is not a handle problem: " << cat.recovery_hint(13);

        // 11 is MAPPING_ERROR, a memory-mapping problem.
        EXPECT_EQ(static_cast<int>(CUBLAS_STATUS_MAPPING_ERROR), 11);
        const std::string_view mapping = cat.recovery_hint(11);
        EXPECT_EQ(mapping.find("internal"), std::string_view::npos) << "a mapping error is not an internal error: " << mapping;
    }

    TEST(CublasErrorTest, StatusNamesAreStable) {
        EXPECT_STREQ(cublas_status_name(CUBLAS_STATUS_SUCCESS), "CUBLAS_STATUS_SUCCESS");
        EXPECT_STREQ(cublas_status_name(CUBLAS_STATUS_NOT_INITIALIZED), "CUBLAS_STATUS_NOT_INITIALIZED");
        EXPECT_STREQ(cublas_status_name(CUBLAS_STATUS_EXECUTION_FAILED), "CUBLAS_STATUS_EXECUTION_FAILED");
        EXPECT_STREQ(cublas_status_name(static_cast<cublasStatus_t>(9999)), "CUBLAS_STATUS_UNKNOWN");
    }

}  // namespace nova::error::test
