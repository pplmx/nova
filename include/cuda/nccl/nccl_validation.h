#pragma once

/**
 * @file nccl_validation.h
 * @brief NCCL prerequisites validation
 *
 * Validates NCCL runtime requirements including version compatibility
 * and shared memory availability. Run these checks before initializing
 * NcclContext to fail early with actionable error messages.
 *
 * @example
 * @code
 * auto result = validate_prerequisites();
 * if (!result) {
 *     std::cerr << "NCCL validation failed: " << result.message << "\n";
 *     for (const auto& warning : result.warnings) {
 *         std::cerr << "  Warning: " << warning << "\n";
 *     }
 * }
 * @endcode
 */

#include <string>
#include <vector>

namespace cuda::nccl {

    /// Minimum NCCL version required (2.25 per STACK.md)
    constexpr int NCCL_MIN_VERSION_MAJOR = 2;
    constexpr int NCCL_MIN_VERSION_MINOR = 25;

    /// Minimum shared memory required for NCCL (512 MB per PITFALLS.md)
    constexpr size_t NCCL_MIN_SHM_BYTES = 512 * 1024 * 1024;

    /**
     * @brief Information about installed NCCL version
     */
    struct VersionInfo {
        /** Major version number */
        int major = 0;

        /** Minor version number */
        int minor = 0;

        /** Patch version number */
        int patch = 0;

        /** Full version string (e.g., "2.25.3") */
        std::string version_string;

        /**
         * @brief Check if version meets minimum requirement
         * @param min_major Minimum major version
         * @param min_minor Minimum minor version
         * @return true if version meets minimum
         */
        [[nodiscard]] bool meets_minimum(int min_major, int min_minor) const {
            if (major > min_major) {
                return true;
            }
            if (major == min_major && minor >= min_minor) {
                return true;
            }
            return false;
        }

        /**
         * @brief Check against NCCL minimum version
         * @return true if version meets NCCL_MIN_VERSION
         */
        [[nodiscard]] bool meets_minimum() const { return meets_minimum(NCCL_MIN_VERSION_MAJOR, NCCL_MIN_VERSION_MINOR); }
    };

    /**
     * @brief Result of a validation check
     */
    struct ValidationResult {
        /** Whether validation passed */
        bool valid = false;

        /** Human-readable result message */
        std::string message;

        /** Additional warnings (non-fatal issues) */
        std::vector<std::string> warnings;

        /** Explicit bool conversion for convenience */
        explicit operator bool() const { return valid; }
    };

    /**
     * @brief Get NCCL version information
     * @return VersionInfo with version details (zeros if NCCL not available)
     */
    VersionInfo get_version();

    /**
     * @brief Check NCCL version at runtime
     *
     * Verifies NCCL version meets minimum requirement (2.25+).
     * Throws exception if version is too old.
     *
     * @param min_major Minimum major version (default: NCCL_MIN_VERSION_MAJOR)
     * @param min_minor Minimum minor version (default: NCCL_MIN_VERSION_MINOR)
     * @return ValidationResult with status
     *
     * @example
     * @code
     * auto result = validate_version();
     * if (!result) {
     *     throw std::runtime_error(result.message);
     * }
     * @endcode
     */
    ValidationResult validate_version(int min_major = NCCL_MIN_VERSION_MAJOR, int min_minor = NCCL_MIN_VERSION_MINOR);

    /**
     * @brief Check shared memory availability for NCCL
     *
     * NCCL uses /dev/shm for inter-process communication.
     * This checks that sufficient space is available.
     *
     * @return ValidationResult with status
     * @note Requires 512MB minimum per PITFALLS.md
     *
     * @example
     * @code
     * auto result = validate_shared_memory();
     * if (!result) {
     *     std::cerr << "Shared memory issue: " << result.message << "\n";
     *     // Docker users should use --shm-size=1g
     * }
     * @endcode
     */
    ValidationResult validate_shared_memory();

    /**
     * @brief Run all validation checks
     *
     * Convenience function that runs all NCCL prerequisites.
     * Use before initializing NcclContext.
     *
     * @return ValidationResult with all issues combined
     *
     * @example
     * @code
     * auto result = validate_prerequisites();
     * if (!result) {
     *     // Fail early with actionable error message
     *     std::cerr << result.message << "\n";
     *     for (const auto& w : result.warnings) {
     *         std::cerr << "  Hint: " << w << "\n";
     *     }
     *     return 1;
     * }
     * @endcode
     */
    ValidationResult validate_prerequisites();

}  // namespace cuda::nccl
