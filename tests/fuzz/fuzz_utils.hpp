#ifndef NOVA_FUZZ_MEMORY_POOL_HPP
#define NOVA_FUZZ_MEMORY_POOL_HPP

#include <cstdint>
#include <cstddef>

namespace nova {
    namespace fuzz {

        class FuzzedDataProvider {
        public:
            FuzzedDataProvider(const uint8_t* data, size_t size)
                : data_(data),
                  size_(size),
                  offset_(0) {}

            template <typename T>
            T ConsumeIntegral() {
                if (remaining() < sizeof(T)) {
                    return T {};
                }
                T value = *reinterpret_cast<const T*>(data_ + offset_);
                offset_ += sizeof(T);
                return value;
            }

            template <typename T>
            T ConsumeIntegralInRange(T min, T max) {
                T value = ConsumeIntegral<T>();
                if (max <= min) {
                    return min;
                }
                return min + (value % (max - min + 1));
            }

            template <typename T>
            T ConsumeFloatingPoint() {
                if (remaining() < sizeof(T)) {
                    return T {};
                }
                T value = *reinterpret_cast<const T*>(data_ + offset_);
                offset_ += sizeof(T);
                return value;
            }

            size_t remaining() const { return size_ > offset_ ? size_ - offset_ : 0; }

        private:
            const uint8_t* data_;
            size_t size_;
            size_t offset_;
        };

    }  // namespace fuzz
}  // namespace nova

#endif  // NOVA_FUZZ_MEMORY_POOL_HPP
