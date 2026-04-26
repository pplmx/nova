#pragma once

#include <cublas_v2.h>
#include <system_error>
#include <string_view>

namespace nova::error {

const char* cublas_status_name(cublasStatus_t status) noexcept;

struct cublas_error_info {
    cublasStatus_t status;
    const char* operation{nullptr};
    const char* file{nullptr};
    int line{0};

    [[nodiscard]] std::string_view message() const noexcept;
    [[nodiscard]] std::string_view recovery_hint() const noexcept;
};

class cublas_error_category : public std::error_category {
public:
    [[nodiscard]] const char* name() const noexcept override { return "cublas"; }
    [[nodiscard]] std::string message(int ev) const override;
    [[nodiscard]] std::string_view recovery_hint(int ev) const noexcept;
};

const std::error_category& cublas_category() noexcept;

inline std::error_code make_error_code(cublasStatus_t status,
                                       const char* operation = nullptr,
                                       const char* file = nullptr,
                                       int line = 0) noexcept {
    return std::error_code(static_cast<int>(status), cublas_category());
}

class cublas_exception : public std::system_error {
public:
    explicit cublas_exception(cublasStatus_t status, const char* operation = nullptr,
                              const char* file = nullptr, int line = 0);

    [[nodiscard]] cublas_error_info info() const noexcept { return info_; }

private:
    cublas_error_info info_;
};

class cublas_error_guard {
public:
    cublas_error_guard(const char* operation, const char* file = nullptr, int line = 0) noexcept;
    ~cublas_error_guard();

    cublas_error_guard(const cublas_error_guard&) = delete;
    cublas_error_guard& operator=(const cublas_error_guard&) = delete;

    void check(cublasStatus_t status);

    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] const cublas_error_info& info() const noexcept { return info_; }

private:
    cublas_error_info info_;
    bool ok_{true};
};

#define CUBLAS_CHECK(call) \
    do { \
        static_assert(sizeof(#call) > 0, "CUBLAS_CHECK requires a cuBLAS call"); \
        nova::error::cublas_error_guard nova_cublas_err_guard{#call, __FILE__, __LINE__}; \
        nova_cublas_err_guard.check(call); \
    } while (0)

} // namespace nova::error
