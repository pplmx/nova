#pragma once

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <vector>
#include <optional>
#include <cuda_runtime.h>

namespace nova::checkpoint {

    class StorageBackend {
    public:
        virtual ~StorageBackend() = default;
        virtual void write(const std::string& path, const void* data, size_t size) = 0;
        virtual void read(const std::string& path, void* data, size_t size) = 0;
        virtual bool exists(const std::string& path) const = 0;

        /**
         * @brief Byte length of a stored blob
         *
         * `read()` requires the caller to know how many bytes to ask for, and the interface offered
         * no way to find out: a caller wanting a whole file of unknown length had to guess a size
         * and retry on failure, which for an in-memory backend that refuses over-long reads never
         * succeeds. Callers ended up hard-coding a bound large enough for the files in question,
         * which is a guess that breaks when they grow.
         *
         * Pure virtual, because a generic implementation cannot be correct: the only thing a
         * backend-independent body could do is probe read lengths, and a probe reports the length it
         * asked for rather than the file's, so a 1000-byte blob comes back as whatever power of two
         * the backend happened to accept (measured: 512 against a map that rejects short reads).
         * A wrong length is worse than a compile error -- callers size buffers with it. Backends
         * must answer it themselves; FileStorageBackend uses the filesystem.
         *
         * @return the blob's length in bytes, or 0 when it does not exist
         */
        virtual size_t size(const std::string& path) = 0;

        virtual std::string resolve_path(const std::string& path) const = 0;
        virtual void remove(const std::string& path) = 0;
        virtual std::vector<std::string> list(const std::string& dir) const = 0;
        virtual void create_directory(const std::string& path) = 0;
    };

    class FileStorageBackend : public StorageBackend {
    public:
        explicit FileStorageBackend(const std::string& base_path);

        void write(const std::string& path, const void* data, size_t size) override;
        void read(const std::string& path, void* data, size_t size) override;
        size_t size(const std::string& path) override;
        bool exists(const std::string& path) const override;
        void remove(const std::string& path) override;
        std::vector<std::string> list(const std::string& dir) const override;
        void create_directory(const std::string& path) override;
        std::string resolve_path(const std::string& path) const override;

        /**
         * @brief Destructor
         *
         * Declared here and defined in the .cpp: Impl is incomplete in this header,
         * so an implicit destructor would make unique_ptr<Impl> unusable and the
         * class unconstructible from other translation units.
         */
        ~FileStorageBackend() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    struct SerializedTensor {
        std::string name;
        std::vector<int64_t> shape;
        int dtype;
        size_t size_bytes;
        std::vector<char> data;
    };

    struct CheckpointManifest {
        int version = 1;
        int step = 0;
        int64_t timestamp = 0;
        std::string model_hash;
        std::string optimizer_hash;
        std::vector<SerializedTensor> model_tensors;
        std::vector<SerializedTensor> optimizer_tensors;
        bool has_rng_state = false;
        std::vector<char> rng_state;
        std::map<std::string, std::string> checksums;
    };

    struct CheckpointInfo {
        int step;
        int64_t timestamp;
        size_t size_bytes;
        std::string path;
    };

    class CheckpointManager {
    public:
        static CheckpointManager& instance();

        void initialize(std::shared_ptr<StorageBackend> backend, cudaStream_t stream);

        void set_checkpoint_interval(int steps);
        void set_max_checkpoints_to_keep(int count);
        void enable_incremental_checkpointing(bool enabled);
        void set_full_checkpoint_interval(int steps);
        void set_auto_checkpoint_on_error(bool enabled);

        void save_checkpoint(int step, const std::map<std::string, std::vector<float>>& model_params, const std::map<std::string, std::vector<float>>& optimizer_states);

        bool load_checkpoint(const std::string& path, std::map<std::string, std::vector<float>>& model_params, std::map<std::string, std::vector<float>>& optimizer_states);

        bool load_latest_checkpoint(std::map<std::string, std::vector<float>>& model_params, std::map<std::string, std::vector<float>>& optimizer_states);

        void on_step_complete(int step);
        void trigger_emergency_checkpoint();

        std::optional<CheckpointInfo> get_checkpoint_info(int step) const;
        std::vector<int> available_checkpoints() const;
        void cleanup_old_checkpoints();

        std::string compute_hash(const void* data, size_t size);

        bool validate_checkpoint(const std::string& path);

    private:
        CheckpointManager() = default;
        ~CheckpointManager() = default;
        CheckpointManager(const CheckpointManager&) = delete;
        CheckpointManager& operator=(const CheckpointManager&) = delete;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace nova::checkpoint
