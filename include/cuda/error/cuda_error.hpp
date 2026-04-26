#pragma once

#include <cuda_runtime.h>
#include <system_error>
#include <string_view>
#include <cstdint>

namespace nova::error {

struct cuda_error_info {
    cudaError_t code;
    const char* operation{nullptr};
    const char* file{nullptr};
    int line{0};
    int device_id{-1};
    void* stream{nullptr};

    [[nodiscard]] std::string message() const noexcept;
    [[nodiscard]] std::string_view recovery_hint() const noexcept;
};

class cuda_error_category : public std::error_category {
public:
    [[nodiscard]] const char* name() const noexcept override { return "cuda"; }
    [[nodiscard]] std::string message(int ev) const override;
    [[nodiscard]] std::string_view recovery_hint(int ev) const noexcept;
};

const std::error_category& cuda_category() noexcept;

inline std::error_code make_error_code(cudaError_t err,
                                       const char* operation = nullptr,
                                       const char* file = nullptr,
                                       int line = 0,
                                       int device = -1,
                                       void* stream = nullptr) noexcept {
    return std::error_code(static_cast<int>(err), cuda_category());
}

class cuda_exception : public std::system_error {
public:
    explicit cuda_exception(cudaError_t err, const char* operation = nullptr,
                            const char* file = nullptr, int line = 0,
                            int device = -1, void* stream = nullptr);

    [[nodiscard]] cuda_error_info info() const noexcept { return info_; }

private:
    cuda_error_info info_;
};

class cuda_error_guard {
public:
    cuda_error_guard(const char* operation, int device = -1, void* stream = nullptr,
                     const char* file = nullptr, int line = 0) noexcept;
    ~cuda_error_guard();

    cuda_error_guard(const cuda_error_guard&) = delete;
    cuda_error_guard& operator=(const cuda_error_guard&) = delete;

    void check(cudaError_t err);

    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] const cuda_error_info& info() const noexcept { return info_; }

private:
    cuda_error_info info_;
    bool ok_{true};
};

#define NOVA_CHECK(call) \
    do { \
        static_assert(sizeof(#call) > 0, "NOVA_CHECK requires a CUDA call"); \
        nova::error::cuda_error_guard nova_err_guard{#call, -1, nullptr, __FILE__, __LINE__}; \
        nova_err_guard.check(call); \
    } while (0)

#define NOVA_CHECK_WITH_STREAM(call, stream) \
    do { \
        nova::error::cuda_error_guard nova_err_guard{#call, -1, stream, __FILE__, __LINE__}; \
        nova_err_guard.check(call); \
    } while (0)

#define CUDA_CHECK(call) NOVA_CHECK(call)

} // namespace nova::error
