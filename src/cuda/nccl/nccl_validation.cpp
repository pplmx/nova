/**
 * @file nccl_validation.cpp
 * @brief NCCL validation implementation
 *
 * Validates NCCL prerequisites including version compatibility
 * and shared memory availability.
 */

#include "cuda/nccl/nccl_validation.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <sys/statfs.h>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

VersionInfo get_version() {
    VersionInfo info;

#if NOVA_NCCL_ENABLED
    int version;
    ncclResult_t result = ncclGetVersion(&version);

    if (result == ncclSuccess) {
        info.major = version / 10000;
        info.minor = (version % 10000) / 100;
        info.patch = version % 100;

        std::ostringstream oss;
        oss << info.major << "." << info.minor << "." << info.patch;
        info.version_string = oss.str();
    }
#endif

    return info;
}

ValidationResult validate_version(int min_major, int min_minor) {
    ValidationResult result;
    result.valid = true;

    VersionInfo version = get_version();

    if (version.major == 0) {
        result.valid = false;
        result.message = "Could not detect NCCL version";
        return result;
    }

    if (!version.meets_minimum(min_major, min_minor)) {
        result.valid = false;
        result.message =
            "NCCL version " + version.version_string +
            " is too old. Minimum required: " +
            std::to_string(min_major) + "." + std::to_string(min_minor) + "+";
        return result;
    }

    result.message = "NCCL version " + version.version_string + " is compatible";

    // Add informational warnings for older versions
    if (version.major < 2 || (version.major == 2 && version.minor < 26)) {
        result.warnings.push_back(
            "NCCL < 2.26 detected. Multi-communicator operations "
            "may require explicit ordering."
        );
    }

    return result;
}

ValidationResult validate_shared_memory() {
    ValidationResult result;
    result.valid = true;

    struct statfs fs_stats;

    if (statfs("/dev/shm", &fs_stats) != 0) {
        result.warnings.push_back(
            "Could not stat /dev/shm -- assuming it is available"
        );
        result.message = "Shared memory check skipped";
        return result;
    }

    size_t shm_available = static_cast<size_t>(fs_stats.f_bavail) *
                           static_cast<size_t>(fs_stats.f_bsize);
    size_t shm_total = static_cast<size_t>(fs_stats.f_blocks) *
                       static_cast<size_t>(fs_stats.f_bsize);

    if (shm_available < NCCL_MIN_SHM_BYTES) {
        result.valid = false;
        result.message =
            "Insufficient shared memory for NCCL: " +
            std::to_string(shm_available / (1024 * 1024)) +
            " MB available, " +
            std::to_string(NCCL_MIN_SHM_BYTES / (1024 * 1024)) +
            " MB required. "
            "NCCL may fail to initialize or hang.";

        result.warnings.push_back(
            "Docker users: Use --shm-size=1g or larger"
        );
        result.warnings.push_back(
            "Or set NCCL_CUMEM_HOST_ENABLE=0 to use cuMem instead"
        );
    } else {
        result.message =
            "Shared memory OK: " +
            std::to_string(shm_available / (1024 * 1024)) + " MB available (" +
            std::to_string(shm_total / (1024 * 1024)) + " MB total)";
    }

    return result;
}

ValidationResult validate_prerequisites() {
    ValidationResult result;
    result.valid = true;

    // Check version
    auto version_result = validate_version();
    if (!version_result) {
        return version_result;
    }
    result.warnings.insert(result.warnings.end(),
                           version_result.warnings.begin(),
                           version_result.warnings.end());

    // Check shared memory
    auto shm_result = validate_shared_memory();
    if (!shm_result) {
        return shm_result;
    }
    result.warnings.insert(result.warnings.end(),
                           shm_result.warnings.begin(),
                           shm_result.warnings.end());

    result.message = "All NCCL prerequisites validated";
    return result;
}

}  // namespace cuda::nccl
