#include <cuda/quantize/qat.hpp>
#include <fstream>
#include <algorithm>

namespace nova {
namespace quantize {

void AMPManager::save_config(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        size_t num_layers = configs_.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));

        for (const auto& pair : configs_) {
            size_t name_len = pair.first.size();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(size_t));
            file.write(pair.first.c_str(), name_len);

            int prec = static_cast<int>(pair.second.precision);
            file.write(reinterpret_cast<const char*>(&prec), sizeof(int));
            file.write(reinterpret_cast<const char*>(&pair.second.scale), sizeof(float));
            file.write(reinterpret_cast<const char*>(&pair.second.zero_point), sizeof(float));
            int trainable = pair.second.trainable ? 1 : 0;
            file.write(reinterpret_cast<const char*>(&trainable), sizeof(int));
        }
    }
}

void AMPManager::load_config(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        configs_.clear();

        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));

        for (size_t i = 0; i < num_layers; ++i) {
            size_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(size_t));

            std::string name(name_len, ' ');
            file.read(&name[0], name_len);

            int prec_int;
            file.read(reinterpret_cast<char*>(&prec_int), sizeof(int));
            Precision prec = static_cast<Precision>(prec_int);

            float scale, zero_point;
            file.read(reinterpret_cast<char*>(&scale), sizeof(float));
            file.read(reinterpret_cast<char*>(&zero_point), sizeof(float));

            int trainable_int;
            file.read(reinterpret_cast<char*>(&trainable_int), sizeof(int));

            LayerConfig config(name, prec);
            config.scale = scale;
            config.zero_point = zero_point;
            config.trainable = (trainable_int != 0);

            configs_[name] = config;
        }
    }
}

} // namespace quantize
} // namespace nova
