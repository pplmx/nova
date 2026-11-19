/**
 * @file nccl_validation_test.cpp
 * @brief Contracts for the NCCL prerequisite checks
 *
 * include/cuda/nccl/nccl_validation.h decides whether the machine can run NCCL
 * at all: version compatibility and /dev/shm headroom. It had no tests.
 *
 * The checks are correct on a healthy host, but one diagnostic is misleading:
 * when the library is compiled WITHOUT NCCL (NOVA_NCCL_ENABLED=0), get_version()
 * correctly returns zeros, while validate_version() reports "Could not detect
 * NCCL version". That message points at a broken installation, when the actual
 * cause is that this build has no NCCL support at all — the fix is different
 * (rebuild with NCCL, not reinstall NCCL).
 */

#include <cuda/nccl/nccl_validation.h>

#include <gtest/gtest.h>

#include <string>

namespace cuda::nccl::test {

    // ------------------------------------------------------- version comparison --

    TEST(VersionInfoTest, MeetsMinimumComparesMajorThenMinor) {
        VersionInfo version;
        version.major = 2;
        version.minor = 25;

        EXPECT_TRUE(version.meets_minimum(2, 25)) << "an exact match meets the minimum";
        EXPECT_TRUE(version.meets_minimum(2, 24));
        EXPECT_FALSE(version.meets_minimum(2, 26));

        // A higher major wins regardless of the minor.
        version.major = 3;
        version.minor = 0;
        EXPECT_TRUE(version.meets_minimum(2, 25));

        // A lower major loses regardless of the minor.
        version.major = 1;
        version.minor = 99;
        EXPECT_FALSE(version.meets_minimum(2, 25));
    }

    TEST(VersionInfoTest, DefaultIsZeroAndDoesNotMeetTheMinimum) {
        const VersionInfo version;

        EXPECT_EQ(version.major, 0);
        EXPECT_EQ(version.minor, 0);
        EXPECT_EQ(version.patch, 0);
        EXPECT_TRUE(version.version_string.empty());
        EXPECT_FALSE(version.meets_minimum()) << "an undetected version must not pass the requirement";
    }

    TEST(VersionInfoTest, RequirementConstantsAreConsistent) {
        // The documented floor is 2.25; a change here should be deliberate.
        EXPECT_EQ(NCCL_MIN_VERSION_MAJOR, 2);
        EXPECT_EQ(NCCL_MIN_VERSION_MINOR, 25);
        EXPECT_EQ(NCCL_MIN_SHM_BYTES, 512u * 1024u * 1024u);
    }

    // ------------------------------------------------------------- get_version --

    TEST(GetVersionTest, ReportsAUsableVersionWhenNcclIsCompiledIn) {
        const VersionInfo version = get_version();

        if (!NOVA_NCCL_ENABLED) {
            EXPECT_EQ(version.major, 0) << "without NCCL there is no version to report";
            EXPECT_TRUE(version.version_string.empty());
            GTEST_SKIP() << "built without NCCL";
        }

        ASSERT_GT(version.major, 0) << "NCCL is compiled in, so a version must resolve";
        // The string is the dotted form of the same numbers.
        const std::string expected = std::to_string(version.major) + "." + std::to_string(version.minor) + "." + std::to_string(version.patch);
        EXPECT_EQ(version.version_string, expected);
    }

    // ---------------------------------------------------------- validate_version --

    TEST(ValidateVersionTest, CompatibleVersionPasses) {
        if (!NOVA_NCCL_ENABLED) {
            GTEST_SKIP() << "built without NCCL";
        }
        const ValidationResult result = validate_version();

        EXPECT_TRUE(result.valid);
        EXPECT_TRUE(static_cast<bool>(result));
        EXPECT_NE(result.message.find("compatible"), std::string::npos) << result.message;
    }

    TEST(ValidateVersionTest, ImpossibleRequirementFailsWithTheNumbers) {
        if (!NOVA_NCCL_ENABLED) {
            GTEST_SKIP() << "built without NCCL";
        }
        const ValidationResult result = validate_version(99, 0);

        EXPECT_FALSE(result.valid);
        EXPECT_FALSE(static_cast<bool>(result));
        // The message must name both what was found and what was required.
        EXPECT_NE(result.message.find("99.0"), std::string::npos) << result.message;
        EXPECT_NE(result.message.find(get_version().version_string), std::string::npos) << result.message;
    }

    // Regression: this said "Could not detect NCCL version", which reads as a broken
    // installation. Without NCCL compiled in there is nothing to detect, and the
    // remedy is a different one.
    TEST(ValidateVersionTest, MissingNcclSaysWhyRatherThanBlamingTheInstall) {
        if (NOVA_NCCL_ENABLED) {
            GTEST_SKIP() << "this build has NCCL";
        }
        const ValidationResult result = validate_version();

        EXPECT_FALSE(result.valid);
        EXPECT_NE(result.message.find("not enabled"), std::string::npos) << "the message should say NCCL support is off in this build: " << result.message;
    }

    TEST(ValidateVersionTest, OlderVersionCarriesTheOrderingWarning) {
        if (!NOVA_NCCL_ENABLED) {
            GTEST_SKIP() << "built without NCCL";
        }
        const VersionInfo version = get_version();
        const ValidationResult result = validate_version();

        ASSERT_TRUE(result.valid);
        if (version.major < 2 || (version.major == 2 && version.minor < 26)) {
            EXPECT_FALSE(result.warnings.empty()) << "NCCL before 2.26 should carry the multi-communicator warning";
        } else {
            EXPECT_TRUE(result.warnings.empty());
        }
    }

    // ------------------------------------------------------ validate_shared_memory --

    TEST(ValidateSharedMemoryTest, ResultIsActionableEitherWay) {
        const ValidationResult result = validate_shared_memory();

        EXPECT_FALSE(result.message.empty());

        if (result.valid) {
            EXPECT_NE(result.message.find("OK"), std::string::npos) << result.message;
        } else {
            // A failure has to say how much is there, how much is needed, and what
            // the operator can do about it.
            EXPECT_NE(result.message.find("MB available"), std::string::npos) << result.message;
            EXPECT_NE(result.message.find("MB required"), std::string::npos) << result.message;
            EXPECT_FALSE(result.warnings.empty()) << "a shared-memory failure must come with remediation hints";
            bool has_docker_hint = false;
            for (const std::string& warning : result.warnings) {
                if (warning.find("shm-size") != std::string::npos) {
                    has_docker_hint = true;
                }
            }
            EXPECT_TRUE(has_docker_hint) << "the Docker --shm-size hint is the usual fix";
        }
    }

    // ------------------------------------------------------- validate_prerequisites --

    TEST(ValidatePrerequisitesTest, AggregatesBothChecks) {
        const ValidationResult version_result = validate_version();
        const ValidationResult shm_result = validate_shared_memory();
        const ValidationResult combined = validate_prerequisites();

        // The aggregate passes only if both of its parts do.
        EXPECT_EQ(combined.valid, version_result.valid && shm_result.valid);

        // Its warnings are at least the union of the parts' warnings.
        EXPECT_GE(combined.warnings.size(), shm_result.warnings.size());
        EXPECT_GE(combined.warnings.size(), version_result.warnings.size());

        EXPECT_FALSE(combined.message.empty());
    }

    TEST(ValidatePrerequisitesTest, FailureMessageComesFromTheFailingCheck) {
        const ValidationResult combined = validate_prerequisites();
        if (combined.valid) {
            GTEST_SKIP() << "this host satisfies every prerequisite";
        }

        const ValidationResult version_result = validate_version();
        const ValidationResult shm_result = validate_shared_memory();

        // Whichever check failed first supplies the message the caller sees.
        const std::string& expected = !version_result.valid ? version_result.message : shm_result.message;
        EXPECT_EQ(combined.message, expected);
    }

}  // namespace cuda::nccl::test
