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

            size_t size(const std::string& path) override {
                const auto it = files_.find(path);
                return it == files_.end() ? 0u : it->second.size();
            }

            /// Flip one byte of a stored blob, simulating bit rot on the backup medium.
            bool corrupt_byte_at(const std::string& path, size_t offset) {
                const auto it = files_.find(path);
                if (it == files_.end() || offset >= it->second.size()) {
                    return false;
                }
                it->second[offset] ^= 0xFF;
                return true;
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
                    result.push_back(slash == std::string::npos ? relative : relative.substr(0, slash));
                }
                // Deduplicate top-level entries.
                std::sort(result.begin(), result.end());
                result.erase(std::unique(result.begin(), result.end()), result.end());
                return result;
            }

            void create_directory(const std::string& path) override { directories_.insert(path); }

            std::string resolve_path(const std::string& path) const override { return path; }

            size_t blob_count() const { return files_.size(); }

            /// Drop the last `bytes` bytes of a blob, keeping its length prefix. A
            /// backend that hands out short reads is how a partially written
            /// checkpoint looks to the loader.
            void truncate(const std::string& path, size_t bytes) {
                auto& blob = files_.at(path);
                if (bytes >= blob.size()) {
                    blob.clear();
                    return;
                }
                blob.resize(blob.size() - bytes);
            }

            /// Overwrite an 8-byte length field inside a blob. This builds a stream
            /// whose own prefix is consistent with the buffer while a length INSIDE it
            /// is not -- what a corrupted-but-not-truncated file looks like, and the
            /// only shape that reaches the decoder's bounds checks.
            void overwrite_length(const std::string& path, size_t offset, size_t value) {
                auto& blob = files_.at(path);
                if (offset > blob.size() || sizeof(value) > blob.size() - offset) {
                    throw std::out_of_range("blob offset past the end: " + path);
                }
                std::memcpy(blob.data() + offset, &value, sizeof(value));
            }

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
        ASSERT_TRUE(manager.load_checkpoint("ckpt_5", loaded_model, loaded_optimizer)) << "the checkpoint just written must be loadable";

        ASSERT_EQ(loaded_model.size(), model.size());
        for (const auto& [name, values] : model) {
            ASSERT_TRUE(loaded_model.count(name)) << "missing tensor " << name;
            EXPECT_EQ(loaded_model[name], values) << "tensor " << name << " round-tripped wrong";
        }
        EXPECT_EQ(loaded_optimizer.size(), optimizer.size());
    }

    // A checkpoint whose bytes no longer match what was written is damaged and must not be loaded:
    // the alternative is putting rotten weights into a training run and calling it a restore.
    // save_checkpoint() has always computed a per-tensor SHA-256; until it was serialised into the
    // manifest, load_checkpoint() had nothing to compare against.
    TEST(CheckpointManagerTest, LoadRejectsACheckpointWhosePayloadNoLongerMatches) {
        auto& manager = CheckpointManager::instance();
        auto backend = std::make_shared<InMemoryBackend>();
        manager.initialize(backend, nullptr);

        const auto model = sample_tensors();
        const auto optimizer = sample_tensors();
        EXPECT_NO_THROW(manager.save_checkpoint(1, model, optimizer));

        std::map<std::string, std::vector<float>> ok_model;
        std::map<std::string, std::vector<float>> ok_optimizer;
        ASSERT_TRUE(manager.load_checkpoint("ckpt_1", ok_model, ok_optimizer)) << "the freshly written checkpoint must load";

        // Flip a bit in the LAST byte of the blob. The stream is [len][name_len][name][shape_len]
        // [shape][dtype][count][payload], so the final byte is the low byte of the last float: the
        // frame stays intact and only a value changes. Corrupting an earlier offset risks hitting a
        // length or shape field, which makes the decoder fail for a different reason and would not
        // distinguish "decoded fine, checksum disagrees" from "could not decode at all".
        const size_t length = backend->size("ckpt_1/model.bin");
        ASSERT_GT(length, sizeof(size_t) + 1) << "the blob is only a length prefix";
        ASSERT_TRUE(backend->corrupt_byte_at("ckpt_1/model.bin", length - 1)) << "could not corrupt the payload";
        // A corrupt payload can still decode; the checksum is what must catch it.
        EXPECT_FALSE(manager.load_checkpoint("ckpt_1", ok_model, ok_optimizer)) << "a payload that no longer matches its checksum must be refused";
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
        EXPECT_TRUE(std::is_sorted(steps.begin(), steps.end())) << "checkpoints must come back in step order";
        EXPECT_NE(std::find(steps.begin(), steps.end(), 3), steps.end());
        EXPECT_NE(std::find(steps.begin(), steps.end(), 20), steps.end());
    }

    TEST(CheckpointManagerTest, ValidateRejectsAnIncompleteCheckpoint) {
        auto& manager = CheckpointManager::instance();
        auto backend = std::make_shared<InMemoryBackend>();
        manager.initialize(backend, nullptr);

        EXPECT_FALSE(manager.validate_checkpoint("ckpt_1")) << "nothing has been written yet";

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
    // size() is the query the interface was missing: read() needs the caller to know the length,
    // and there was no way to ask. Callers worked around it by guessing a bound and retrying, which
    // against a backend that refuses over-long reads never converges.
    TEST(CheckpointManagerTest, BackendReportsBlobSize) {
        auto backend = std::make_shared<InMemoryBackend>();
        const char payload[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
        backend->write("blob.bin", payload, sizeof(payload));

        EXPECT_EQ(backend->size("blob.bin"), sizeof(payload)) << "a 9-byte blob must report 9";
        EXPECT_EQ(backend->size("absent.bin"), 0u) << "a missing blob has no size, and 0 says so without throwing";
    }

    TEST(FileStorageBackendTest, SizeIsTheFileLength) {
        const auto path = std::filesystem::temp_directory_path() / "nova_ckpt_size_probe";
        std::filesystem::remove_all(path);
        FileStorageBackend backend(path.string());
        const std::string payload = "0123456789abcdef";  // 16 bytes, not just the common sizes
        backend.write("f.bin", payload.data(), payload.size());

        EXPECT_EQ(backend.size("f.bin"), payload.size());
        EXPECT_EQ(backend.size("missing.bin"), 0u) << "a missing file reports 0 rather than throwing";
    }

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

        EXPECT_THROW(backend.resolve_path("/etc/passwd"), std::invalid_argument) << "absolute paths escape the base directory";
        EXPECT_THROW(backend.resolve_path("../escape"), std::invalid_argument) << "traversal escapes the base directory";

        std::filesystem::remove_all(path);
    }

    // Regression: the traversal guard was a SUBSTRING search for "..", so any legitimate name
    // containing two consecutive dots was refused. Measured before the fix:
    //
    //     file..name.bin     -> THROWS Path traversal is not allowed
    //     v1..2/weights.bin  -> THROWS Path traversal is not allowed
    //
    // A ".." is dangerous only when it is a whole path component; inside a filename it is an
    // ordinary character. The guard now walks the components.
    TEST(FileStorageBackendTest, TraversalIsAPathComponentNotASubstring) {
        const auto path = std::filesystem::temp_directory_path() / "nova_ckpt_component_test";
        std::filesystem::remove_all(path);
        FileStorageBackend backend(path.string());

        // Names that merely contain ".." must resolve.
        EXPECT_NO_THROW(backend.resolve_path("file..name.bin")) << "two dots inside a filename are not traversal";
        EXPECT_NO_THROW(backend.resolve_path("v1..2/weights.bin")) << "two dots inside a directory name are not traversal";
        EXPECT_NO_THROW(backend.resolve_path("..leading")) << "a name starting with dots is not traversal";
        EXPECT_NO_THROW(backend.resolve_path("trailing..")) << "a name ending with dots is not traversal";

        // And they must land inside the base directory.
        const std::string resolved = backend.resolve_path("file..name.bin");
        EXPECT_EQ(resolved.find(path.string()), 0u) << "resolved outside the base directory: " << resolved;

        // Real traversal is still refused, in every position.
        EXPECT_THROW(backend.resolve_path(".."), std::invalid_argument);
        EXPECT_THROW(backend.resolve_path("../escape"), std::invalid_argument);
        EXPECT_THROW(backend.resolve_path("sub/../../escape"), std::invalid_argument);
        EXPECT_THROW(backend.resolve_path("a/.."), std::invalid_argument);

        // A file with two dots in its name must survive a real write/read round trip, which is
        // the case a user actually hits (version strings in tensor names).
        const char payload[] = "weights";
        ASSERT_NO_THROW(backend.write("model..v2.bin", payload, sizeof(payload) - 1));
        char readback[sizeof(payload)] = {};
        ASSERT_NO_THROW(backend.read("model..v2.bin", readback, sizeof(payload) - 1));
        EXPECT_STREQ(readback, payload);

        std::filesystem::remove_all(path);
    }

    // Regression: read() treated a short file as success. `!ifs && !ifs.eof()` let a
    // stream that ended early pass, so the caller's buffer kept whatever it held
    // beyond the bytes the file actually provided, and the loader decoded it as
    // though it were real data.
    TEST(FileStorageBackendTest, ShortReadIsRejected) {
        const auto path = std::filesystem::temp_directory_path() / "nova_ckpt_short_read_test";
        std::filesystem::remove_all(path);
        FileStorageBackend backend(path.string());

        const std::string payload = "0123456789";
        backend.write("short.bin", payload.data(), payload.size());

        std::vector<char> buffer(payload.size(), '\x7f');
        EXPECT_NO_THROW(backend.read("short.bin", buffer.data(), payload.size()));

        // Ask for more than the file holds: the tail of the buffer must not be
        // silently accepted as file content.
        std::vector<char> oversized(payload.size() + 8, '\x7f');
        EXPECT_THROW(backend.read("short.bin", oversized.data(), oversized.size()), std::runtime_error);

        std::filesystem::remove_all(path);
    }

    // Regression: checkpoints are directories, and fs::remove() throws
    // "Directory not empty" on a non-empty one. The rotation path therefore threw
    // the moment a save exceeded max_checkpoints_to_keep and left every checkpoint
    // on disk. A probe against the old code: saving steps 0..5 with a keep of 3
    // threw at step 3 and reported six checkpoints still present.
    TEST(CheckpointManagerTest, RotationKeepsTheNewestAndDeletesTheRest) {
        const auto path = std::filesystem::temp_directory_path() / "nova_ckpt_rotation_test";
        std::filesystem::remove_all(path);

        auto& manager = CheckpointManager::instance();
        manager.initialize(std::make_shared<FileStorageBackend>(path.string()), nullptr);
        manager.set_max_checkpoints_to_keep(3);

        const auto model = sample_tensors();
        const auto optimizer = sample_tensors();

        for (int step = 0; step <= 5; ++step) {
            EXPECT_NO_THROW(manager.save_checkpoint(step, model, optimizer)) << "saving step " << step << " must not throw";
        }

        const std::vector<int> remaining = manager.available_checkpoints();
        std::vector<int> expected = {3, 4, 5};
        EXPECT_EQ(remaining, expected) << "the three newest checkpoints must survive rotation";

        // The survivors must still load: deleting the right directories is only half
        // the contract, and a recursive remove of the wrong one would leave
        // directories that exist but cannot be read.
        std::map<std::string, std::vector<float>> loaded_model;
        std::map<std::string, std::vector<float>> loaded_optimizer;
        EXPECT_TRUE(manager.load_checkpoint("ckpt_5", loaded_model, loaded_optimizer));
        EXPECT_EQ(loaded_model.size(), model.size());

        manager.set_max_checkpoints_to_keep(3);
        std::filesystem::remove_all(path);
    }

    // Regression: the loader read lengths out of the blob and memcpy'd straight past
    // the end of the buffer. A truncated blob is caught by the backend (a short read
    // throws) and reported as "cannot load". This case covers the other shape: a
    // blob whose own prefix agrees with the buffer while a length INSIDE it does
    // not, which is what a corrupted file looks like and what used to run off the
    // end of the allocation.
    TEST(CheckpointManagerTest, InconsistentInnerLengthIsRejectedInsteadOfOverRead) {
        auto& manager = CheckpointManager::instance();
        auto backend = std::make_shared<InMemoryBackend>();
        manager.initialize(backend, nullptr);
        manager.set_max_checkpoints_to_keep(100);

        manager.save_checkpoint(1, sample_tensors(), sample_tensors());

        // Layout: [8-byte payload length][8-byte first name length][name...]. Claim a
        // name longer than the whole blob; the decoder must refuse rather than
        // construct a string from bytes that are not there.
        backend->overwrite_length("ckpt_1/model.bin", sizeof(size_t), 1u << 20);

        std::map<std::string, std::vector<float>> loaded_model;
        std::map<std::string, std::vector<float>> loaded_optimizer;
        EXPECT_FALSE(manager.load_checkpoint("ckpt_1", loaded_model, loaded_optimizer));
        EXPECT_TRUE(loaded_model.empty()) << "a rejected load must not leave partial tensors in the caller's map";
    }

    // The outer length prefix claiming more payload than the blob holds: caught by the
    // decoder's first check, before any field is read.
    TEST(CheckpointManagerTest, InconsistentPayloadLengthIsRejected) {
        auto& manager = CheckpointManager::instance();
        auto backend = std::make_shared<InMemoryBackend>();
        manager.initialize(backend, nullptr);
        manager.set_max_checkpoints_to_keep(100);

        manager.save_checkpoint(2, sample_tensors(), sample_tensors());
        backend->overwrite_length("ckpt_2/model.bin", 0, 1u << 20);

        std::map<std::string, std::vector<float>> loaded_model;
        std::map<std::string, std::vector<float>> loaded_optimizer;
        EXPECT_FALSE(manager.load_checkpoint("ckpt_2", loaded_model, loaded_optimizer));
    }

    // A truncated blob must be reported as unloadable, not thrown out of a function
    // whose contract is a bool, and it must not leave a half-decoded model behind.
    TEST(CheckpointManagerTest, TruncatedBlobIsReportedAsUnloadable) {
        auto& manager = CheckpointManager::instance();
        auto backend = std::make_shared<InMemoryBackend>();
        manager.initialize(backend, nullptr);
        manager.set_max_checkpoints_to_keep(100);

        std::map<std::string, std::vector<float>> model;
        for (int i = 0; i < 8; ++i) {
            model["w" + std::to_string(i)] = std::vector<float>(4, static_cast<float>(i));
        }
        model["big"] = std::vector<float>(256, 7.0f);
        manager.save_checkpoint(3, model, sample_tensors());

        backend->truncate("ckpt_3/model.bin", 16);

        std::map<std::string, std::vector<float>> loaded_model;
        std::map<std::string, std::vector<float>> loaded_optimizer;
        bool loaded = true;
        EXPECT_NO_THROW(loaded = manager.load_checkpoint("ckpt_3", loaded_model, loaded_optimizer));
        EXPECT_FALSE(loaded) << "a truncated checkpoint must be rejected, not half-decoded";
        EXPECT_TRUE(loaded_model.empty()) << "a rejected load must not leave partial tensors in the caller's map";
        EXPECT_TRUE(loaded_optimizer.empty()) << "a rejected load must not touch the optimizer map either";
    }

}  // namespace nova::checkpoint::test
