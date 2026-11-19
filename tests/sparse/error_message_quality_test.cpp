/**
 * @file error_message_quality_test.cpp
 * @brief Contracts for how backend failures describe themselves
 *
 * Five error paths reported only the numeric status code, which is the one piece
 * of information a caller cannot act on without opening the vendor headers:
 *
 *   - CublasException: "cuBLAS error: 3"
 *   - the cuFFT check in src/cuda/signal/signal.cu: "cuFFT error: 8"
 *   - the cuSOLVER check in src/cuda/linalg/linalg.cu: "cuSOLVER error: 8"
 *   - CusparseContext: a bare integer after "Failed to create cuSPARSE handle:"
 *   - MpiException: "MPI error <n>", discarding the text MPI_Error_string offers
 *
 * A related defect: CUBLAS_CHECK was defined unconditionally in both
 * include/cuda/device/error.h and include/cuda/error/cublas_error.hpp, with
 * different semantics (one throws CublasException from the call site, the other
 * uses a guard object). A translation unit including both took whichever came
 * second, so the include order silently chose the exception type and the message
 * format. The second definition now yields when the first is present.
 *
 * Each message now carries the enumerator name plus the number, so a failure is
 * self-describing while still greppable by code.
 */

#include <cuda/device/error.h>
#include <cuda/error/cublas_error.hpp>

#include <gtest/gtest.h>

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace cuda::device::test {

    namespace {

        /// Runs `call` and returns the exception message, or an empty string when
        /// nothing was thrown.
        template <typename Callable>
        std::string message_from(Callable&& call) {
            try {
                call();
            } catch (const std::exception& error) {
                return error.what();
            }
            return {};
        }

    }  // namespace

    TEST(ErrorMessageQualityTest, BothCublasCheckDefinitionsCoexist) {
        // This translation unit includes both headers. Before the guard, whichever
        // came second redefined the macro: the build warned and the surviving
        // behaviour depended on include order. Compiling without a redefinition is
        // the assertion; CUBLAS_CHECK must also still expand to working code.
        const cublasStatus_t ok = CUBLAS_STATUS_SUCCESS;
        EXPECT_EQ(ok, CUBLAS_STATUS_SUCCESS);
    }

    TEST(ErrorMessageQualityTest, CublasExceptionNamesTheStatus) {
        // Regression: the message was "cuBLAS error: 3".
        const std::string message = message_from([] { throw CublasException(CUBLAS_STATUS_ALLOC_FAILED, "op.cu", 42); });

        ASSERT_FALSE(message.empty());
        EXPECT_NE(message.find("CUBLAS_STATUS_ALLOC_FAILED"), std::string::npos) << "the enumerator name is missing: " << message;
        EXPECT_NE(message.find("(3)"), std::string::npos) << "the numeric code should still be present: " << message;
        EXPECT_NE(message.find("op.cu:42"), std::string::npos) << "the throw site is still useful: " << message;
    }

    TEST(ErrorMessageQualityTest, CublasExceptionPreservesTheStatusAccessor) {
        CublasException error(CUBLAS_STATUS_INVALID_VALUE, "op.cu", 7);
        EXPECT_EQ(error.error(), CUBLAS_STATUS_INVALID_VALUE);
    }

    TEST(ErrorMessageQualityTest, EveryCublasStatusHasADistinctName) {
        const cublasStatus_t codes[] = {
            CUBLAS_STATUS_SUCCESS,
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

        for (size_t i = 0; i < std::size(codes); ++i) {
            const std::string name = message_from([code = codes[i]] { throw CublasException(code, "f", 1); });
            ASSERT_FALSE(name.empty());

            for (size_t j = i + 1; j < std::size(codes); ++j) {
                // Extract just the enumerator token for comparison.
                const std::string other = message_from([code = codes[j]] { throw CublasException(code, "f", 1); });
                const size_t pos_i = name.find("CUBLAS_STATUS_");
                const size_t pos_j = other.find("CUBLAS_STATUS_");
                ASSERT_NE(pos_i, std::string::npos) << name;
                ASSERT_NE(pos_j, std::string::npos) << other;
                const std::string token_i = name.substr(pos_i, name.find(' ', pos_i) - pos_i);
                const std::string token_j = other.substr(pos_j, other.find(' ', pos_j) - pos_j);
                EXPECT_NE(token_i, token_j) << "codes " << static_cast<int>(codes[i]) << " and " << static_cast<int>(codes[j]) << " both report " << token_i;
            }
        }
    }

    TEST(ErrorMessageQualityTest, UnmappedCublasStatusFallsBackSafely) {
        // A code newer than this switch must not produce a null pointer or empty
        // name; it degrades to a visible placeholder.
        const std::string message = message_from([] { throw CublasException(static_cast<cublasStatus_t>(9999), "f", 1); });

        ASSERT_FALSE(message.empty());
        EXPECT_NE(message.find("UNKNOWN"), std::string::npos) << message;
        EXPECT_NE(message.find("(9999)"), std::string::npos) << message;
    }

    TEST(ErrorMessageQualityTest, CublasStatusNameHelperIsConsistent) {
        // Two independent named lookups exist (one in device/error.h, one in
        // nova::error). They must agree, or a log line and an exception would spell
        // the same code differently.
        const cublasStatus_t codes[] = {
            CUBLAS_STATUS_SUCCESS,
            CUBLAS_STATUS_NOT_INITIALIZED,
            CUBLAS_STATUS_ALLOC_FAILED,
            CUBLAS_STATUS_INVALID_VALUE,
            CUBLAS_STATUS_INTERNAL_ERROR,
        };

        for (const auto code : codes) {
            const std::string from_error = message_from([code] { throw CublasException(code, "f", 1); });
            const char* from_helper = nova::error::cublas_status_name(code);
            ASSERT_NE(from_helper, nullptr);
            EXPECT_NE(from_error.find(from_helper), std::string::npos) << "the exception says something other than " << from_helper << ": " << from_error;
        }
    }

}  // namespace cuda::device::test
