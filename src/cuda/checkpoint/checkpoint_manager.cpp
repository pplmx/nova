#include "cuda/checkpoint/checkpoint_manager.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <openssl/sha.h>

namespace nova::checkpoint {

namespace fs = std::filesystem;

struct FileStorageBackend::Impl {
    std::string base_path;

    std::string resolve_path(const std::string& path) const {
        fs::path p = fs::path(path);
        if (p.is_absolute()) {
            throw std::invalid_argument("Absolute paths are not allowed: " + path);
        }
        std::string normalized = path;
        if (normalized.find("..") != std::string::npos) {
            throw std::invalid_argument("Path traversal is not allowed: " + path);
        }
        if (path.empty()) {
            return base_path;
        }
        return (fs::path(base_path) / path).string();
    }

    void atomic_write(const std::string& full_path, const void* data, size_t size) {
        auto now = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        std::string temp_path = full_path + ".tmp." + std::to_string(getpid()) + "." + std::to_string(ns);
        std::ofstream ofs(temp_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open file for writing: " + temp_path);
        }
        ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        ofs.close();
        if (!ofs) {
            fs::remove(temp_path);
            throw std::runtime_error("Failed to write file: " + temp_path);
        }
        fs::rename(temp_path, full_path);
    }
};

FileStorageBackend::FileStorageBackend(const std::string& base_path)
    : impl_(std::make_unique<Impl>()) {
    impl_->base_path = base_path;
    if (!fs::exists(impl_->base_path)) {
        fs::create_directories(impl_->base_path);
    }
}

void FileStorageBackend::write(const std::string& path, const void* data, size_t size) {
    auto full_path = impl_->resolve_path(path);
    auto dir = fs::path(full_path).parent_path();
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    impl_->atomic_write(full_path, data, size);
}

void FileStorageBackend::read(const std::string& path, void* data, size_t size) {
    auto full_path = impl_->resolve_path(path);
    std::ifstream ifs(full_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + full_path);
    }
    ifs.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!ifs && !ifs.eof()) {
        throw std::runtime_error("Failed to read file: " + full_path);
    }
}

bool FileStorageBackend::exists(const std::string& path) const {
    return fs::exists(impl_->resolve_path(path));
}

void FileStorageBackend::remove(const std::string& path) {
    fs::remove(impl_->resolve_path(path));
}

std::vector<std::string> FileStorageBackend::list(const std::string& dir) const {
    std::vector<std::string> result;
    auto full_path = impl_->resolve_path(dir);
    if (!fs::exists(full_path) || !fs::is_directory(full_path)) {
        return result;
    }
    for (const auto& entry : fs::directory_iterator(full_path)) {
        result.push_back(entry.path().filename());
    }
    return result;
}

void FileStorageBackend::create_directory(const std::string& path) {
    fs::create_directories(impl_->resolve_path(path));
}

std::string FileStorageBackend::resolve_path(const std::string& path) const {
    return impl_->resolve_path(path);
}

struct CheckpointManager::Impl {
    std::shared_ptr<StorageBackend> backend;
    cudaStream_t stream = nullptr;
    int checkpoint_interval = 1000;
    int max_checkpoints_to_keep = 3;
    bool incremental_enabled = false;
    int full_checkpoint_interval = 10;
    bool auto_checkpoint_on_error = true;
    int last_checkpoint_step = -1;
    std::string latest_checkpoint_path;
};

CheckpointManager& CheckpointManager::instance() {
    static CheckpointManager manager;
    return manager;
}

void CheckpointManager::initialize(std::shared_ptr<StorageBackend> backend, cudaStream_t stream) {
    impl_->backend = std::move(backend);
    impl_->stream = stream;
}

void CheckpointManager::set_checkpoint_interval(int steps) {
    impl_->checkpoint_interval = steps;
}

void CheckpointManager::set_max_checkpoints_to_keep(int count) {
    impl_->max_checkpoints_to_keep = count;
}

void CheckpointManager::enable_incremental_checkpointing(bool enabled) {
    impl_->incremental_enabled = enabled;
}

void CheckpointManager::set_full_checkpoint_interval(int steps) {
    impl_->full_checkpoint_interval = steps;
}

void CheckpointManager::set_auto_checkpoint_on_error(bool enabled) {
    impl_->auto_checkpoint_on_error = enabled;
}

std::string CheckpointManager::compute_hash(const void* data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char*>(data), size, hash);
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

void CheckpointManager::save_checkpoint(
    int step,
    const std::map<std::string, std::vector<float>>& model_params,
    const std::map<std::string, std::vector<float>>& optimizer_states) {

    auto& impl = *impl_;

    std::string ckpt_dir = "/ckpt_" + std::to_string(step);
    impl.backend->create_directory(ckpt_dir);

    CheckpointManifest manifest;
    manifest.step = step;
    manifest.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    for (const auto& [name, data] : model_params) {
        SerializedTensor tensor;
        tensor.name = name;
        tensor.shape = {static_cast<int64_t>(data.size())};
        tensor.dtype = CUDA_R_32F;
        tensor.size_bytes = data.size() * sizeof(float);
        tensor.data.resize(tensor.size_bytes);
        std::memcpy(tensor.data.data(), data.data(), tensor.size_bytes);

        std::string checksum = compute_hash(tensor.data.data(), tensor.size_bytes);
        manifest.checksums[name] = checksum;

        manifest.model_tensors.push_back(std::move(tensor));
    }

    for (const auto& [name, data] : optimizer_states) {
        SerializedTensor tensor;
        tensor.name = name;
        tensor.shape = {static_cast<int64_t>(data.size())};
        tensor.dtype = CUDA_R_32F;
        tensor.size_bytes = data.size() * sizeof(float);
        tensor.data.resize(tensor.size_bytes);
        std::memcpy(tensor.data.data(), data.data(), tensor.size_bytes);
        manifest.optimizer_tensors.push_back(std::move(tensor));
    }

    std::string manifest_json = std::to_string(manifest.version) + "\n" +
                                std::to_string(manifest.step) + "\n" +
                                std::to_string(manifest.timestamp) + "\n" +
                                std::to_string(manifest.model_tensors.size()) + "\n" +
                                std::to_string(manifest.optimizer_tensors.size());

    impl.backend->write(ckpt_dir + "/manifest.txt", manifest_json.data(), manifest_json.size());

    size_t offset = 0;
    std::vector<char> model_file;
    for (const auto& tensor : manifest.model_tensors) {
        size_t name_size = tensor.name.size();
        model_file.resize(model_file.size() + sizeof(size_t));
        std::memcpy(model_file.data() + offset, &name_size, sizeof(size_t));
        offset += sizeof(size_t);

        model_file.insert(model_file.end(), tensor.name.begin(), tensor.name.end());
        offset += name_size;

        size_t shape_size = tensor.shape.size() * sizeof(int64_t);
        model_file.resize(model_file.size() + sizeof(size_t));
        std::memcpy(model_file.data() + offset, &shape_size, sizeof(size_t));
        offset += sizeof(size_t);

        model_file.resize(model_file.size() + shape_size);
        std::memcpy(model_file.data() + offset, tensor.shape.data(), shape_size);
        offset += shape_size;

        model_file.resize(model_file.size() + sizeof(int));
        std::memcpy(model_file.data() + offset, &tensor.dtype, sizeof(int));
        offset += sizeof(int);

        model_file.resize(model_file.size() + tensor.size_bytes);
        std::memcpy(model_file.data() + offset, tensor.data.data(), tensor.size_bytes);
        offset += tensor.size_bytes;
    }

    size_t model_total_size = model_file.size();
    std::vector<char> model_blob(sizeof(size_t) + model_file.size());
    std::memcpy(model_blob.data(), &model_total_size, sizeof(size_t));
    std::memcpy(model_blob.data() + sizeof(size_t), model_file.data(), model_file.size());
    impl.backend->write(ckpt_dir + "/model.bin", model_blob.data(), model_blob.size());

    offset = 0;
    std::vector<char> optimizer_file;
    for (const auto& tensor : manifest.optimizer_tensors) {
        size_t name_size = tensor.name.size();
        optimizer_file.resize(optimizer_file.size() + sizeof(size_t));
        std::memcpy(optimizer_file.data() + offset, &name_size, sizeof(size_t));
        offset += sizeof(size_t);

        optimizer_file.insert(optimizer_file.end(), tensor.name.begin(), tensor.name.end());
        offset += name_size;

        size_t shape_size = tensor.shape.size() * sizeof(int64_t);
        optimizer_file.resize(optimizer_file.size() + sizeof(size_t));
        std::memcpy(optimizer_file.data() + offset, &shape_size, sizeof(size_t));
        offset += sizeof(size_t);

        optimizer_file.resize(optimizer_file.size() + shape_size);
        std::memcpy(optimizer_file.data() + offset, tensor.shape.data(), shape_size);
        offset += shape_size;

        optimizer_file.resize(optimizer_file.size() + sizeof(int));
        std::memcpy(optimizer_file.data() + offset, &tensor.dtype, sizeof(int));
        offset += sizeof(int);

        optimizer_file.resize(optimizer_file.size() + tensor.size_bytes);
        std::memcpy(optimizer_file.data() + offset, tensor.data.data(), tensor.size_bytes);
        offset += tensor.size_bytes;
    }

    size_t optimizer_total_size = optimizer_file.size();
    std::vector<char> optimizer_blob(sizeof(size_t) + optimizer_file.size());
    std::memcpy(optimizer_blob.data(), &optimizer_total_size, sizeof(size_t));
    std::memcpy(optimizer_blob.data() + sizeof(size_t), optimizer_file.data(), optimizer_file.size());
    impl.backend->write(ckpt_dir + "/optimizer.bin", optimizer_blob.data(), optimizer_blob.size());

    impl.last_checkpoint_step = step;
    impl.latest_checkpoint_path = ckpt_dir;

    cleanup_old_checkpoints();
}

bool CheckpointManager::load_checkpoint(
    const std::string& path,
    std::map<std::string, std::vector<float>>& model_params,
    std::map<std::string, std::vector<float>>& optimizer_states) {

    auto& impl = *impl_;

    if (!impl.backend->exists(path + "/model.bin") ||
        !impl.backend->exists(path + "/optimizer.bin")) {
        return false;
    }

    std::vector<char> model_file;
    {
        size_t model_size = 0;
        std::ifstream ifs(impl.backend->resolve_path(path + "/model.bin"), std::ios::binary);
        if (ifs) {
            ifs.seekg(0, std::ios::end);
            model_file.resize(ifs.tellg());
            ifs.seekg(0);
            ifs.read(model_file.data(), model_file.size());
        }
    }

    if (model_file.size() < sizeof(size_t)) {
        return false;
    }

    size_t data_size;
    std::memcpy(&data_size, model_file.data(), sizeof(size_t));
    size_t offset = sizeof(size_t);
    size_t model_end = sizeof(size_t) + data_size;
    while (offset < model_end) {
        size_t name_size;
        std::memcpy(&name_size, model_file.data() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        std::string name(model_file.data() + offset, name_size);
        offset += name_size;

        size_t shape_size;
        std::memcpy(&shape_size, model_file.data() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        std::vector<int64_t> shape(shape_size / sizeof(int64_t));
        std::memcpy(shape.data(), model_file.data() + offset, shape_size);
        offset += shape_size;

        int dtype;
        std::memcpy(&dtype, model_file.data() + offset, sizeof(int));
        offset += sizeof(int);

        size_t tensor_size = 1;
        for (auto dim : shape) {
            tensor_size *= dim;
        }

        std::vector<float> data(tensor_size);
        std::memcpy(data.data(), model_file.data() + offset, tensor_size * sizeof(float));
        offset += tensor_size * sizeof(float);

        model_params[name] = std::move(data);
    }

    std::vector<char> optimizer_file;
    {
        std::ifstream ifs(impl.backend->resolve_path(path + "/optimizer.bin"), std::ios::binary);
        if (ifs) {
            ifs.seekg(0, std::ios::end);
            optimizer_file.resize(ifs.tellg());
            ifs.seekg(0);
            ifs.read(optimizer_file.data(), optimizer_file.size());
        }
    }

    if (optimizer_file.size() < sizeof(size_t)) {
        return false;
    }

    size_t opt_data_size;
    std::memcpy(&opt_data_size, optimizer_file.data(), sizeof(size_t));
    offset = sizeof(size_t);
    size_t opt_end = sizeof(size_t) + opt_data_size;
    while (offset < opt_end) {
        size_t name_size;
        std::memcpy(&name_size, optimizer_file.data() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        std::string name(optimizer_file.data() + offset, name_size);
        offset += name_size;

        size_t shape_size;
        std::memcpy(&shape_size, optimizer_file.data() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        std::vector<int64_t> shape(shape_size / sizeof(int64_t));
        std::memcpy(shape.data(), optimizer_file.data() + offset, shape_size);
        offset += shape_size;

        int dtype;
        std::memcpy(&dtype, optimizer_file.data() + offset, sizeof(int));
        offset += sizeof(int);

        size_t tensor_size = 1;
        for (auto dim : shape) {
            tensor_size *= dim;
        }

        std::vector<float> data(tensor_size);
        std::memcpy(data.data(), optimizer_file.data() + offset, tensor_size * sizeof(float));
        offset += tensor_size * sizeof(float);

        optimizer_states[name] = std::move(data);
    }

    return true;
}

bool CheckpointManager::load_latest_checkpoint(
    std::map<std::string, std::vector<float>>& model_params,
    std::map<std::string, std::vector<float>>& optimizer_states) {

    auto& impl = *impl_;

    if (impl.latest_checkpoint_path.empty()) {
        return false;
    }

    return load_checkpoint(impl.latest_checkpoint_path, model_params, optimizer_states);
}

void CheckpointManager::on_step_complete(int step) {
    auto& impl = *impl_;

    if (impl.last_checkpoint_step >= 0 &&
        step - impl.last_checkpoint_step < impl.checkpoint_interval) {
        return;
    }

    std::map<std::string, std::vector<float>> dummy_model;
    std::map<std::string, std::vector<float>> dummy_optimizer;
    save_checkpoint(step, dummy_model, dummy_optimizer);
}

void CheckpointManager::trigger_emergency_checkpoint() {
    auto& impl = *impl_;

    if (!impl.auto_checkpoint_on_error) {
        return;
    }

    std::map<std::string, std::vector<float>> dummy_model;
    std::map<std::string, std::vector<float>> dummy_optimizer;

    int emergency_step = (impl.last_checkpoint_step >= 0)
                         ? impl.last_checkpoint_step + 1
                         : 0;
    save_checkpoint(emergency_step, dummy_model, dummy_optimizer);
}

std::optional<CheckpointInfo> CheckpointManager::get_checkpoint_info(int step) const {
    auto& impl = *impl_;

    std::string ckpt_dir = "/ckpt_" + std::to_string(step);
    if (!impl.backend->exists(ckpt_dir)) {
        return std::nullopt;
    }

    CheckpointInfo info;
    info.step = step;
    info.path = ckpt_dir;

    return info;
}

std::vector<int> CheckpointManager::available_checkpoints() const {
    auto& impl = *impl_;
    std::vector<int> checkpoints;

    auto entries = impl.backend->list("/");
    for (const auto& entry : entries) {
        if (entry.rfind("ckpt_", 0) == 0) {
            try {
                int step = std::stoi(entry.substr(5));
                checkpoints.push_back(step);
            } catch (...) {
            }
        }
    }

    std::sort(checkpoints.begin(), checkpoints.end());
    return checkpoints;
}

void CheckpointManager::cleanup_old_checkpoints() {
    auto& impl = *impl_;

    auto checkpoints = available_checkpoints();
    while (static_cast<int>(checkpoints.size()) > impl.max_checkpoints_to_keep) {
        int to_remove = checkpoints.front();
        std::string ckpt_dir = "/ckpt_" + std::to_string(to_remove);
        impl.backend->remove(ckpt_dir);
        checkpoints.erase(checkpoints.begin());
    }
}

bool CheckpointManager::validate_checkpoint(const std::string& path) {
    auto& impl = *impl_;

    return impl.backend->exists(path + "/manifest.txt") &&
           impl.backend->exists(path + "/model.bin") &&
           impl.backend->exists(path + "/optimizer.bin");
}

} // namespace nova::checkpoint
