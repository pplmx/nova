#pragma once

/**
 * @file test_temp_paths.h
 * @brief Portable process id and scratch paths for tests.
 *
 * Tests used <unistd.h> for ::getpid() and hardcoded "/tmp". Neither exists on
 * Windows: MSVC has no <unistd.h>, and there is no /tmp. Both are only used to
 * build a unique filename, so a portable equivalent is all the tests need.
 */

#include <filesystem>
#include <string>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace nova::test {

    /// Current process id as int. _getpid() on Windows, getpid() elsewhere.
    inline int process_id() {
#if defined(_WIN32)
        return static_cast<int>(::_getpid());
#else
        return static_cast<int>(::getpid());
#endif
    }

    /// Directory to write scratch files to. Tests previously hardcoded /tmp,
    /// which does not exist on Windows.
    inline std::filesystem::path temp_dir() {
        return std::filesystem::temp_directory_path();
    }

    /// A unique scratch file path: <temp_dir>/<prefix><pid>_<suffix>.
    inline std::string temp_path(const std::string& prefix, const std::string& suffix) {
        return (temp_dir() / (prefix + std::to_string(process_id()) + suffix)).string();
    }

}  // namespace nova::test
