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
    virtual void remove(const std::string& path) = 0;
    virtual std::vector<std::string> list(const std::string& dir) const = 0;
    virtual void create_directory(const std::string& path) = 0;
    virtual std::string resolve_path(const std::string& path) const = 0;
};

class FileStorageBackend : public StorageBackend {
public:
    explicit FileStorageBackend(const std::string& base_path);

    void write(const std::string& path, const void* data, size_t size) override;
    void read(const std::string& path, void* data, size_t size) override;
    bool exists(const std::string& path) const override;
    void remove(const std::string& path) override;
    std::vector<std::string> list(const std::string& dir) const override;
    void create_directory(const std::string& path) override;
    std::string resolve_path(const std::string& path) const override;

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

    void save_checkpoint(int step,
                         const std::map<std::string, std::vector<float>>& model_params,
                         const std::map<std::string, std::vector<float>>& optimizer_states);

    bool load_checkpoint(const std::string& path,
                         std::map<std::string, std::vector<float>>& model_params,
                         std::map<std::string, std::vector<float>>& optimizer_states);

    bool load_latest_checkpoint(std::map<std::string, std::vector<float>>& model_params,
                                std::map<std::string, std::vector<float>>& optimizer_states);

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

} // namespace nova::checkpoint
