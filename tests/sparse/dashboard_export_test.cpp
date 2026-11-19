/**
 * @file dashboard_export_test.cpp
 * @brief The dashboard generator must not report success it did not achieve
 *
 * src/cuda/performance/dashboard/dashboard_exporter.cpp does not implement the C++
 * dashboard export. generate_html() returns a placeholder string naming the Python
 * generator that does, and write_files() was an empty body returning true:
 *
 *     bool DashboardGenerator::write_files(const std::string& output_dir) const {
 *         return true;
 *     }
 *
 * Measured before the fix: write_files("/tmp/nova_dash_probe") returned true and left
 * the output directory non-existent. A caller that checked the flag concluded a
 * dashboard had been written.
 *
 * write_files now returns false. These cases pin that, and pin the placeholder, so a
 * future implementation has to change them deliberately rather than silently flipping a
 * boolean.
 */

#include <cuda/performance/dashboard/dashboard_exporter.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace cuda::performance::dashboard::test {

    namespace {

        /// A scratch directory this case owns outright.
        std::string scratch_directory() {
            return (std::filesystem::temp_directory_path() / "nova_dashboard_export_test").string();
        }

    }  // namespace

    // Regression: this returned true without creating anything.
    TEST(DashboardExportTest, WriteFilesDoesNotClaimSuccess) {
        const std::string directory = scratch_directory();
        std::filesystem::remove_all(directory);

        DashboardGenerator generator;
        const bool written = generator.write_files(directory);

        EXPECT_FALSE(written) << "the C++ export is not implemented, so it must not report success";

        std::filesystem::remove_all(directory);
    }

    TEST(DashboardExportTest, ReturningFalseMatchesTheFilesystem) {
        // The flag and reality must agree. Whichever way write_files answers, the directory
        // it was given has to exist exactly when it said so.
        const std::string directory = scratch_directory();
        std::filesystem::remove_all(directory);

        DashboardGenerator generator;
        const bool written = generator.write_files(directory);
        const bool exists = std::filesystem::exists(directory);

        EXPECT_EQ(written, exists) << "write_files said " << (written ? "success" : "failure") << " but the directory " << (exists ? "exists" : "does not exist");

        std::filesystem::remove_all(directory);
    }

    TEST(DashboardExportTest, TheHtmlPlaceholderNamesTheAlternative) {
        // The placeholder is the only signal a caller gets about where the real generator
        // lives, so it must keep naming it.
        DashboardGenerator generator;
        const std::string html = generator.generate_html();

        EXPECT_FALSE(html.empty());
        EXPECT_NE(html.find("not implemented"), std::string::npos) << "the placeholder should say the C++ path is unimplemented: " << html;
        EXPECT_NE(html.find("generate_performance_dashboard.py"), std::string::npos) << "the placeholder should name the Python generator: " << html;
    }

    TEST(DashboardExportTest, AnEmptyOutputDirectoryIsAlsoReportedAsFailure) {
        // Rejecting an empty path is not the reason this returns false — the export is
        // absent in every case — but a caller passing nothing must not be told otherwise.
        DashboardGenerator generator;
        EXPECT_FALSE(generator.write_files(""));
    }

}  // namespace cuda::performance::dashboard::test
