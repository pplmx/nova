#ifndef NOVA_FUZZ_MEMORY_POOL_HPP
#define NOVA_FUZZ_MEMORY_POOL_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>

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
                // memcpy, not a reinterpret_cast: offset_ advances by whatever the
                // preceding ConsumeX() calls consumed, so data_ + offset_ carries no
                // alignment guarantee. Dereferencing a T* there is undefined behaviour and
                // UBSan reports it on every call --
                //
                //   runtime error: load of misaligned address ... for type
                //   'const long unsigned int', which requires 8 byte alignment
                //
                // which is harmless on x86 and a fault on the strict-alignment targets this
                // library also supports. memcpy of sizeof(T) bytes is defined for any
                // alignment and compiles to the same single load.
                T value {};
                std::memcpy(&value, data_ + offset_, sizeof(T));
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
