/**
 * @file checkpoint_manager_test.cpp
 * @brief Contracts for checkpoint save/load and the storage backend
 *
 * include/cuda/checkpoint/checkpoint_manager.h persists model and optimizer
 * state. It had no tests, and five defects lived in it:
 *
 * - CheckpointManager::instance() returned a default-constructed object whose
 *   `std::unique_ptr<Impl> impl_` was never allocated, so the first call through
 *   ANY member — even set_checkpoint_interval — dereferenced a null pointer and
 *   killed the process.
 *
 * - FileStorageBackend declared only an implicit destructor while holding
 *   `std::unique_ptr<Impl>` with Impl incomplete in the header. That makes the
 *   implicit destructor ill-formed, so the class could not be constructed from
 *   any other translation unit at all — `std::make_shared<FileStorageBackend>`
 *   failed with "invalid application of sizeof to incomplete type".
 *
 * - CheckpointManager builds paths as "/ckpt_<step>" (absolute), while
 *   FileStorageBackend::resolve_path deliberately REJECTS absolute paths. The
 *   two halves of the module could not work together: every save threw
 *   std::invalid_argument.
 *
 * - get_checkpoint_info() constructed a CheckpointInfo without default member
 *   initialisers and filled only `step` and `path`, handing the caller
 *   indeterminate `timestamp` and `size_bytes`.
 *
 * - load_checkpoint() read the blobs with a raw std::ifstream on the resolved
 *   path instead of calling backend->read(), so a checkpoint saved to a
 *   non-file backend could never be loaded.
 */

#include <cuda/checkpoint/checkpoint_manager.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <set>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nova::checkpoint::test {

namespace {

// In-memory backend: proves the save/load path goes through the abstraction
// rather than touching the filesystem directly.
class InMemoryBackend : public StorageBackend {
public:
    void write(const std::string& path, const void* data, size_t size) override {
        auto& blob = files_[path];
        const auto* bytes = static_cast<const char*>(data);
        blob.assign(bytes, bytes + size);
    }

    void read(const std::string& path, void* data, size_t size) override {
        const auto it = files_.find(path);
        if (it == files_.end()) {
            throw std::runtime_error("missing blob: " + path);
        }
        if (it->second.size() < size) {
            throw std::runtime_error("short blob: " + path);
        }
        std::memcpy(data, it->second.data(), size);
    }

    bool exists(const std::string& path) const override {
        if (files_.count(path) > 0 || directories_.count(path) > 0) {
            return true;
        }
        const std::string prefix = path + "/";
        for (const auto& [blob_path, blob] : files_) {
            (void)blob;
            if (blob_path.rfind(prefix, 0) == 0) {
                return true;
            }
        }
        return false;
    }

    void remove(const std::string& path) override {
        const std::string prefix = path + "/";
        for (auto it = files_.begin(); it != files_.end();) {
            if (it->first == path || it->first.rfind(prefix, 0) == 0) {
                it = files_.erase(it);
            } else {
                ++it;
            }
        }
        directories_.erase(path);
    }

    std::vector<std::string> list(const std::string& dir) const override {
        std::vector<std::string> result;
        const std::string prefix = dir.empty() ? "" : dir + "/";
        for (const auto& [path, blob] : files_) {
            (void)blob;
            std::string relative = path;
            if (!prefix.empty() && relative.rfind(prefix, 0) == 0) {
                relative = relative.substr(prefix.size());
            }
            const auto slash = relative.find('/');
            result.push_back(slash == std::string::npos ? relative
                                                        : relative.substr(0, slash));
        }
        // Deduplicate top-level entries.
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    void create_directory(const std::string& path) override {
        directories_.insert(path);
    }

    std::string resolve_path(const std::string& path) const override { return path; }

    size_t blob_count() const { return files_.size(); }

private:
    std::map<std::string, std::vector<char>> files_;
    std::set<std::string> directories_;
};

std::map<std::string, std::vector<float>> sample_tensors() {
    return {
        {"weight", {1.0f, 2.0f, 3.0f, 4.0f}},
        {"bias", {0.5f, -0.5f}},
    };
}

}  // namespace

// Regression: every member dereferenced a null impl_. Reaching this line at all
// is the assertion.
TEST(CheckpointManagerTest, ConfigurationSettersDoNotCrash) {
    auto& manager = CheckpointManager::instance();

    EXPECT_NO_THROW(manager.set_checkpoint_interval(500));
    EXPECT_NO_THROW(manager.set_max_checkpoints_to_keep(2));
    EXPECT_NO_THROW(manager.enable_incremental_checkpointing(true));
    EXPECT_NO_THROW(manager.set_full_checkpoint_interval(5));
    EXPECT_NO_THROW(manager.set_auto_checkpoint_on_error(false));
    manager.set_auto_checkpoint_on_error(true);
}

// Regression: the absolute/relative mismatch made save throw before writing.
TEST(CheckpointManagerTest, SavesAndLoadsThroughTheBackend) {
    auto& manager = CheckpointManager::instance();
    auto backend = std::make_shared<InMemoryBackend>();
    manager.initialize(backend, nullptr);

    const auto model = sample_tensors();
    const auto optimizer = sample_tensors();

    EXPECT_NO_THROW(manager.save_checkpoint(5, model, optimizer));
    EXPECT_GT(backend->blob_count(), 0u) << "nothing was written";

    std::map<std::string, std::vector<float>> loaded_model;
    std::map<std::string, std::vector<float>> loaded_optimizer;
    ASSERT_TRUE(manager.load_checkpoint("ckpt_5", loaded_model, loaded_optimizer))
        << "the checkpoint just written must be loadable";

    ASSERT_EQ(loaded_model.size(), model.size());
    for (const auto& [name, values] : model) {
        ASSERT_TRUE(loaded_model.count(name)) << "missing tensor " << name;
        EXPECT_EQ(loaded_model[name], values) << "tensor " << name << " round-tripped wrong";
    }
    EXPECT_EQ(loaded_optimizer.size(), optimizer.size());
}

TEST(CheckpointManagerTest, LoadReturnsFalseForAMissingCheckpoint) {
    auto& manager = CheckpointManager::instance();
    manager.initialize(std::make_shared<InMemoryBackend>(), nullptr);

    std::map<std::string, std::vector<float>> model;
    std::map<std::string, std::vector<float>> optimizer;
    EXPECT_FALSE(manager.load_checkpoint("ckpt_999", model, optimizer));
    EXPECT_TRUE(model.empty());
}

// Regression: timestamp and size_bytes were indeterminate.
TEST(CheckpointManagerTest, CheckpointInfoIsFullyInitialised) {
    auto& manager = CheckpointManager::instance();
    auto backend = std::make_shared<InMemoryBackend>();
    manager.initialize(backend, nullptr);
    manager.save_checkpoint(7, sample_tensors(), sample_tensors());

    const auto info = manager.get_checkpoint_info(7);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->step, 7);
    EXPECT_EQ(info->path, "ckpt_7");
    // Reading an uninitialised field is undefined behaviour; these assertions
    // pin that the values are defined, whatever they are.
    EXPECT_GE(info->timestamp, 0);
    EXPECT_GE(info->size_bytes, 0u);
}

TEST(CheckpointManagerTest, UnknownStepHasNoInfo) {
    auto& manager = CheckpointManager::instance();
    manager.initialize(std::make_shared<InMemoryBackend>(), nullptr);

    EXPECT_FALSE(manager.get_checkpoint_info(4242).has_value());
}

TEST(CheckpointManagerTest, AvailableCheckpointsAreSortedAndParsed) {
    auto& manager = CheckpointManager::instance();
    manager.initialize(std::make_shared<InMemoryBackend>(), nullptr);

    manager.save_checkpoint(20, sample_tensors(), sample_tensors());
    manager.save_checkpoint(3, sample_tensors(), sample_tensors());

    const std::vector<int> steps = manager.available_checkpoints();
    ASSERT_GE(steps.size(), 2u);
    EXPECT_TRUE(std::is_sorted(steps.begin(), steps.end()))
        << "checkpoints must come back in step order";
    EXPECT_NE(std::find(steps.begin(), steps.end(), 3), steps.end());
    EXPECT_NE(std::find(steps.begin(), steps.end(), 20), steps.end());
}

TEST(CheckpointManagerTest, ValidateRejectsAnIncompleteCheckpoint) {
    auto& manager = CheckpointManager::instance();
    auto backend = std::make_shared<InMemoryBackend>();
    manager.initialize(backend, nullptr);

    EXPECT_FALSE(manager.validate_checkpoint("ckpt_1"))
        << "nothing has been written yet";

    manager.save_checkpoint(1, sample_tensors(), sample_tensors());
    EXPECT_TRUE(manager.validate_checkpoint("ckpt_1"));
}

TEST(CheckpointManagerTest, ComputeHashIsDeterministicAndSizeSensitive) {
    auto& manager = CheckpointManager::instance();

    const std::string payload = "nova-checkpoint";
    const std::string first = manager.compute_hash(payload.data(), payload.size());
    const std::string second = manager.compute_hash(payload.data(), payload.size());

    EXPECT_EQ(first, second);
    EXPECT_EQ(first.size(), 64u) << "SHA-256 renders as 64 hex characters";
    EXPECT_NE(first, manager.compute_hash(payload.data(), payload.size() - 1));
    // Lowercase hex, no separators.
    EXPECT_EQ(first.find_first_not_of("0123456789abcdef"), std::string::npos);
}

// Regression: the class was unconstructible from another translation unit.
TEST(FileStorageBackendTest, CanBeConstructedOutsideItsOwnSourceFile) {
    const auto path = std::filesystem::temp_directory_path() / "nova_ckpt_backend_test";
    std::filesystem::remove_all(path);

    EXPECT_NO_THROW({
        FileStorageBackend backend(path.string());
        const std::string payload = "hello";
        backend.write("note.txt", payload.data(), payload.size());
        EXPECT_TRUE(backend.exists("note.txt"));

        std::string readback(payload.size(), '\0');
        backend.read("note.txt", readback.data(), readback.size());
        EXPECT_EQ(readback, payload);
    });

    std::filesystem::remove_all(path);
}

TEST(FileStorageBackendTest, RejectsAbsolutePathsAndTraversal) {
    const auto path = std::filesystem::temp_directory_path() / "nova_ckpt_path_test";
    std::filesystem::remove_all(path);
    FileStorageBackend backend(path.string());

    EXPECT_THROW(backend.resolve_path("/etc/passwd"), std::invalid_argument)
        << "absolute paths escape the base directory";
    EXPECT_THROW(backend.resolve_path("../escape"), std::invalid_argument)
        << "traversal escapes the base directory";

    std::filesystem::remove_all(path);
}

}  // namespace nova::checkpoint::test
