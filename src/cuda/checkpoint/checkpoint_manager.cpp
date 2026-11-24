#include "cuda/checkpoint/checkpoint_manager.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
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
            // Traversal is a PATH COMPONENT, not a substring. This used to reject any path
            // containing the two characters "..", which refused legitimate names such as
            // "file..name.bin" and "v1..2/weights.bin" -- measured, both were rejected while a
            // plain "sub/model.bin" resolved normally. Walk the components instead and reject
            // only a component that is exactly "..".
            for (const auto& component : p) {
                if (component == "..") {
                    throw std::invalid_argument("Path traversal is not allowed: " + path);
                }
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

    FileStorageBackend::~FileStorageBackend() = default;

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
        // A short read is a failure, not an end of file to be tolerated. `!ifs &&
        // !ifs.eof()` let a file shorter than the caller asked for return
        // successfully with the tail of `data` untouched, so a truncated blob
        // reached the decoder looking like a complete one. Both failure modes now
        // throw: reading less than was promised means the caller's buffer is not
        // what the file says it should be.
        if (!ifs && !ifs.eof()) {
            throw std::runtime_error("Failed to read file: " + full_path);
        }
        if (ifs.eof() && size > 0 && static_cast<size_t>(ifs.gcount()) < size) {
            throw std::runtime_error("Short read from " + full_path + ": wanted " + std::to_string(size) + " bytes, got " + std::to_string(static_cast<size_t>(ifs.gcount())));
        }
    }

    size_t FileStorageBackend::size(const std::string& path) {
        const auto full_path = impl_->resolve_path(path);
        std::error_code ec;
        const auto bytes = fs::file_size(full_path, ec);
        return ec ? 0u : static_cast<size_t>(bytes);
    }

    bool FileStorageBackend::exists(const std::string& path) const {
        return fs::exists(impl_->resolve_path(path));
    }

    void FileStorageBackend::remove(const std::string& path) {
        // Checkpoints are DIRECTORIES holding model.bin, optimizer.bin and
        // manifest.txt. fs::remove() on a non-empty directory throws
        // ("Directory not empty"), so the rotate-old-checkpoints path could never
        // remove one: saving past max_checkpoints_to_keep threw and left every
        // checkpoint on disk. remove() is the backend's recursive delete, which is
        // what a checkpoint directory needs.
        auto full_path = impl_->resolve_path(path);
        std::error_code ec;
        fs::remove_all(full_path, ec);
        if (ec) {
            throw std::runtime_error("Failed to remove " + full_path + ": " + ec.message());
        }
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
        // impl_ is a unique_ptr that nothing allocated — the constructor is
        // `= default` — so the first call through any member (even
        // set_checkpoint_interval) dereferenced a null pointer and killed the
        // process.
        static CheckpointManager manager;
        // A lambda cannot capture a function-local static by reference, so the
        // object is passed in explicitly instead.
        static const bool initialized = [](auto& instance) {
            instance.impl_ = std::make_unique<CheckpointManager::Impl>();
            return true;
        }(manager);
        (void)initialized;
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

    void CheckpointManager::save_checkpoint(int step,
                                            const std::map<std::string, std::vector<float>>& model_params,
                                            const std::map<std::string, std::vector<float>>& optimizer_states) {
        auto& impl = *impl_;

        std::string ckpt_dir = "ckpt_" + std::to_string(step);
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
            manifest.checksums["model/" + name] = checksum;

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
            manifest.checksums["optimizer/" + name] = compute_hash(tensor.data.data(), tensor.size_bytes);
            manifest.optimizer_tensors.push_back(std::move(tensor));
        }

        std::string manifest_json = std::to_string(manifest.version) + "\n" + std::to_string(manifest.step) + "\n" + std::to_string(manifest.timestamp) + "\n"
                                    + std::to_string(manifest.model_tensors.size()) + "\n" + std::to_string(manifest.optimizer_tensors.size());
        // The per-tensor SHA-256 lived only in this local object: computed, stored in a map, and
        // dropped when the function returned, so nothing on disk ever recorded what the payload was
        // supposed to hash to and load_checkpoint() had nothing to check. Written out as a trailing
        // section of "<key> <hex>" lines, keyed by section because a model tensor and an optimizer
        // entry may share a name. The five leading lines are unchanged, so a reader that stops after
        // them is unaffected.
        for (const auto& [key, hex] : manifest.checksums) {
            manifest_json += "\n" + key + " " + hex;
        }

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

    namespace {

        /// Parse the "<key> <hex>" checksum lines save_checkpoint() appends to the manifest.
        std::map<std::string, std::string> read_manifest_checksums(StorageBackend& backend, const std::string& path) {
            std::string text;
            const size_t bytes = backend.size(path);
            if (bytes > 0) {
                std::vector<char> buffer(bytes);
                try {
                    backend.read(path, buffer.data(), buffer.size());
                    text.assign(buffer.begin(), buffer.end());
                } catch (const std::exception&) {
                    return {};
                }
            }
            if (text.empty()) {
                return {};
            }

            std::map<std::string, std::string> checksums;
            std::istringstream lines(text);
            std::string line;
            int index = 0;
            while (std::getline(lines, line)) {
                if (index++ < 5) {
                    continue;  // version, step, timestamp, model count, optimizer count
                }
                const auto space = line.find(' ');
                if (space == std::string::npos || space + 1 >= line.size()) {
                    continue;
                }
                checksums[line.substr(0, space)] = line.substr(space + 1);
            }
            return checksums;
        }

        /// True when `count` bytes at `offset` are inside `blob`.
        bool in_bounds(const std::vector<char>& blob, size_t offset, size_t count) {
            return offset <= blob.size() && count <= blob.size() - offset;
        }

        /**
         * Decode the tensor stream written by save_checkpoint() into `out`.
         *
         * Every field is a length or a count taken from the file, so each one is a
         * promise the file makes about bytes that follow. This validated each read
         * against the buffer. It did not before: a blob truncated mid-tensor — a
         * killed process, a full disk, a partial copy — had its lengths read and
         * then `memcpy`'d straight past the end of the buffer. AddressSanitizer on a
         * checkpoint truncated to half its length reports
         *
         *     READ of size 4 ... heap-buffer-overflow
         *     checkpoint_manager.cpp:349 in CheckpointManager::load_checkpoint
         *
         * and on a shorter truncation the same code silently returned true with a
         * tensor built from whatever followed the allocation. A checkpoint that
         * cannot be decoded must be rejected, not half-applied.
         *
         * @return true when the whole stream decoded, false when it is malformed.
         */
        bool decode_tensor_stream(const std::vector<char>& blob, std::map<std::string, std::vector<float>>& out) {
            if (blob.size() < sizeof(size_t)) {
                return false;
            }

            size_t data_size = 0;
            std::memcpy(&data_size, blob.data(), sizeof(size_t));
            // The prefix is what save_checkpoint() wrote as the payload length, so a
            // stream claiming more bytes than the blob holds is inconsistent even
            // before the first field is read.
            if (!in_bounds(blob, sizeof(size_t), data_size)) {
                return false;
            }

            size_t offset = sizeof(size_t);
            const size_t end = sizeof(size_t) + data_size;

            while (offset < end) {
                size_t name_size = 0;
                if (!in_bounds(blob, offset, sizeof(size_t))) {
                    return false;
                }
                std::memcpy(&name_size, blob.data() + offset, sizeof(size_t));
                offset += sizeof(size_t);
                if (!in_bounds(blob, offset, name_size)) {
                    return false;
                }
                std::string name(blob.data() + offset, name_size);
                offset += name_size;

                size_t shape_size = 0;
                if (!in_bounds(blob, offset, sizeof(size_t))) {
                    return false;
                }
                std::memcpy(&shape_size, blob.data() + offset, sizeof(size_t));
                offset += sizeof(size_t);
                if (shape_size % sizeof(int64_t) != 0 || !in_bounds(blob, offset, shape_size)) {
                    return false;
                }
                std::vector<int64_t> shape(shape_size / sizeof(int64_t));
                if (shape_size > 0) {
                    std::memcpy(shape.data(), blob.data() + offset, shape_size);
                }
                offset += shape_size;

                int dtype = 0;
                if (!in_bounds(blob, offset, sizeof(int))) {
                    return false;
                }
                std::memcpy(&dtype, blob.data() + offset, sizeof(int));
                offset += sizeof(int);

                // The element count is a product of dimensions read from the file, so
                // it can overflow before it is ever compared against the buffer. Detect
                // the wrap rather than computing a small count from a huge shape.
                size_t elements = 1;
                for (const auto dim : shape) {
                    if (dim < 0) {
                        return false;
                    }
                    const auto udim = static_cast<size_t>(dim);
                    if (udim != 0 && elements > std::numeric_limits<size_t>::max() / udim) {
                        return false;
                    }
                    elements *= udim;
                }
                if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
                    return false;
                }
                const size_t bytes = elements * sizeof(float);
                if (!in_bounds(blob, offset, bytes)) {
                    return false;
                }
                std::vector<float> data(elements);
                if (bytes > 0) {
                    std::memcpy(data.data(), blob.data() + offset, bytes);
                }
                offset += bytes;

                out[name] = std::move(data);
            }

            return offset == end;
        }

    }  // namespace

    bool CheckpointManager::load_checkpoint(const std::string& path,
                                            std::map<std::string, std::vector<float>>& model_params,
                                            std::map<std::string, std::vector<float>>& optimizer_states) {
        auto& impl = *impl_;

        if (!impl.backend->exists(path + "/model.bin") || !impl.backend->exists(path + "/optimizer.bin")) {
            return false;
        }

        // Read through the backend, not a raw ifstream: the whole point of the
        // StorageBackend abstraction is that a checkpoint may live somewhere other
        // than the local filesystem. Bypassing it here meant anything saved to a
        // non-file backend could never be loaded.
        std::vector<char> model_file;
        std::vector<char> optimizer_file;
        try {
            // The backend owns the blobs; ask it how big they are by reading the
            // 8-byte length prefix, then read the whole thing. Probing with a raw
            // std::ifstream made every non-filesystem backend unloadable.
            //
            // The backend throws on a short or unreadable blob, which for a
            // checkpoint that exists but is damaged is the same answer as a malformed
            // stream: it cannot be loaded. Turning that into `false` here keeps the
            // documented contract of this function (a bool) instead of letting a
            // backend-specific exception escape into the caller's training loop.
            size_t model_size = 0;
            impl.backend->read(path + "/model.bin", &model_size, sizeof(size_t));
            model_file.resize(sizeof(size_t) + model_size);
            impl.backend->read(path + "/model.bin", model_file.data(), model_file.size());

            size_t optimizer_size = 0;
            impl.backend->read(path + "/optimizer.bin", &optimizer_size, sizeof(size_t));
            optimizer_file.resize(sizeof(size_t) + optimizer_size);
            impl.backend->read(path + "/optimizer.bin", optimizer_file.data(), optimizer_file.size());
        } catch (const std::exception&) {
            return false;
        }

        // Decode into temporaries and only then move them into the caller's maps. A
        // checkpoint that fails half way through used to leave the model half
        // replaced and the optimizer untouched, which looks like a successful load
        // that silently lost tensors.
        std::map<std::string, std::vector<float>> decoded_model;
        std::map<std::string, std::vector<float>> decoded_optimizer;
        if (!decode_tensor_stream(model_file, decoded_model) || !decode_tensor_stream(optimizer_file, decoded_optimizer)) {
            return false;
        }

        // Verify the decoded payload against the checksums save_checkpoint() records. Before they
        // were written to disk there was nothing to verify -- the hash was computed into a local map
        // and dropped -- so a checkpoint whose bits had rotted loaded successfully. Measured:
        // flipping the low byte of the last float in model.bin leaves the stream frame perfectly
        // decodable and used to return true.
        //
        // A manifest with no checksum lines is accepted: those were written by an older build and
        // carry nothing to compare against, and refusing them would break restores that worked
        // before this change. One that has them must match.
        const auto expected = read_manifest_checksums(*impl.backend, path + "/manifest.txt");
        if (!expected.empty()) {
            const auto matches = [&](const std::map<std::string, std::vector<float>>& tensors, const char* section) {
                for (const auto& [name, values] : tensors) {
                    const auto it = expected.find(std::string(section) + "/" + name);
                    if (it != expected.end() && compute_hash(values.data(), values.size() * sizeof(float)) != it->second) {
                        return false;
                    }
                }
                return true;
            };
            if (!matches(decoded_model, "model") || !matches(decoded_optimizer, "optimizer")) {
                return false;
            }
        }

        model_params = std::move(decoded_model);
        optimizer_states = std::move(decoded_optimizer);
        return true;
    }

    bool CheckpointManager::load_latest_checkpoint(std::map<std::string, std::vector<float>>& model_params, std::map<std::string, std::vector<float>>& optimizer_states) {
        auto& impl = *impl_;

        if (impl.latest_checkpoint_path.empty()) {
            return false;
        }

        return load_checkpoint(impl.latest_checkpoint_path, model_params, optimizer_states);
    }

    void CheckpointManager::on_step_complete(int step) {
        auto& impl = *impl_;

        if (impl.last_checkpoint_step >= 0 && step - impl.last_checkpoint_step < impl.checkpoint_interval) {
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

        int emergency_step = (impl.last_checkpoint_step >= 0) ? impl.last_checkpoint_step + 1 : 0;
        save_checkpoint(emergency_step, dummy_model, dummy_optimizer);
    }

    std::optional<CheckpointInfo> CheckpointManager::get_checkpoint_info(int step) const {
        auto& impl = *impl_;

        std::string ckpt_dir = "ckpt_" + std::to_string(step);
        if (!impl.backend->exists(ckpt_dir)) {
            return std::nullopt;
        }

        // Every field is set: CheckpointInfo has no default member initialisers, so
        // returning a partially filled struct handed the caller indeterminate
        // timestamp and size_bytes (undefined behaviour on read).
        CheckpointInfo info {};
        info.step = step;
        info.path = ckpt_dir;
        // The timestamp save_checkpoint() writes into the manifest, read back. It used to be a
        // hard-coded 0: the manifest carried the real value and nothing consumed it, so a caller
        // showing or sorting checkpoints by age saw every one of them dated to the epoch. The
        // existing test asserted `timestamp >= 0`, which 0 satisfies, so nothing noticed.
        info.timestamp = 0;
        {
            const std::string manifest_path = ckpt_dir + "/manifest.txt";
            const size_t manifest_size = impl.backend->size(manifest_path);
            if (manifest_size > 0) {
                std::vector<char> buffer(manifest_size);
                try {
                    impl.backend->read(manifest_path, buffer.data(), buffer.size());
                    std::istringstream lines(std::string(buffer.begin(), buffer.end()));
                    std::string line;
                    int index = 0;
                    while (std::getline(lines, line)) {
                        if (index == 2) {  // version, step, timestamp
                            try {
                                info.timestamp = std::stoll(line);
                            } catch (const std::exception&) {
                                info.timestamp = 0;
                            }
                            break;
                        }
                        ++index;
                    }
                } catch (const std::exception&) {
                    info.timestamp = 0;
                }
            }
        }
        // Size comes from the length prefix the save path writes, so this works for
        // any backend rather than only ones backed by a real filesystem.
        size_t model_size = 0;
        if (impl.backend->exists(ckpt_dir + "/model.bin")) {
            impl.backend->read(ckpt_dir + "/model.bin", &model_size, sizeof(size_t));
        }
        size_t optimizer_size = 0;
        if (impl.backend->exists(ckpt_dir + "/optimizer.bin")) {
            impl.backend->read(ckpt_dir + "/optimizer.bin", &optimizer_size, sizeof(size_t));
        }
        info.size_bytes = model_size + optimizer_size;
        return info;
    }

    std::vector<int> CheckpointManager::available_checkpoints() const {
        auto& impl = *impl_;
        std::vector<int> checkpoints;

        auto entries = impl.backend->list("");
        for (const auto& entry : entries) {
            if (entry.rfind("ckpt_", 0) == 0) {
                try {
                    int step = std::stoi(entry.substr(5));
                    checkpoints.push_back(step);
                } catch (...) {
                    // A directory entry that starts with "ckpt_" but is not followed by a
                    // number is not a checkpoint. Skipping it is the point of iterating; the
                    // alternative — failing the listing because one unrelated entry exists —
                    // would make the whole feature unusable in a directory anyone else can
                    // write to.
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
            std::string ckpt_dir = "ckpt_" + std::to_string(to_remove);
            impl.backend->remove(ckpt_dir);
            checkpoints.erase(checkpoints.begin());
        }
    }

    bool CheckpointManager::validate_checkpoint(const std::string& path) {
        auto& impl = *impl_;

        return impl.backend->exists(path + "/manifest.txt") && impl.backend->exists(path + "/model.bin") && impl.backend->exists(path + "/optimizer.bin");
    }

}  // namespace nova::checkpoint
