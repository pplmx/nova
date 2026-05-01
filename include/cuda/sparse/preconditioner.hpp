#pragma once

#include "cuda/sparse/matrix.hpp"
#include "cuda/memory/buffer.h"
#include <memory>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace nova::sparse {

namespace memory = cuda::memory;

class PreconditionerError : public std::runtime_error {
public:
    explicit PreconditionerError(const std::string& msg) : std::runtime_error(msg) {}
};

template<typename T>
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    virtual void setup(const SparseMatrix<T>& A) = 0;

    virtual void apply(const T* in, T* out) = 0;

    virtual void apply(const memory::Buffer<T>& in, memory::Buffer<T>& out) = 0;
};

template<typename T>
class JacobiPreconditioner : public Preconditioner<T> {
public:
    explicit JacobiPreconditioner(T omega = T{1.0})
        : omega_(omega) {
        if (omega <= T{0} || omega > T{2}) {
            throw PreconditionerError(
                "JacobiPreconditioner: omega must be in (0, 2], got " + std::to_string(static_cast<double>(omega)));
        }
    }

    void setup(const SparseMatrix<T>& A) override {
        const int n = A.rows();

        diagonal_.resize(n);
        std::vector<T> h_diagonal(n);

        for (int i = 0; i < n; ++i) {
            const int row_start = A.row_offsets()[i];
            const int row_end = A.row_offsets()[i + 1];
            T diag_val = T{0};

            for (int idx = row_start; idx < row_end; ++idx) {
                if (A.col_indices()[idx] == i) {
                    diag_val = A.values()[idx];
                    break;
                }
            }

            if (std::abs(diag_val) < std::numeric_limits<T>::epsilon()) {
                throw PreconditionerError(
                    "JacobiPreconditioner: zero (or near-zero) diagonal entry at row " +
                    std::to_string(i) + ". Consider using a different preconditioner or matrix reordering.");
            }

            h_diagonal[i] = T{1.0} / diag_val;
        }

        diagonal_.copy_from(h_diagonal.data(), n);
    }

    void apply(const T* in, T* out) override {
        const int n = diagonal_.size();
        std::vector<T> h_in(n), h_out(n);
        std::copy(in, in + n, h_in.begin());

        for (int i = 0; i < n; ++i) {
            h_out[i] = omega_ * diagonal_.data()[i] * h_in[i];
        }

        std::copy(h_out.begin(), h_out.end(), out);
    }

    void apply(const memory::Buffer<T>& in, memory::Buffer<T>& out) override {
        const int n = diagonal_.size();
        out.resize(n);

        std::vector<T> h_in(n), h_out(n);
        in.copy_to(h_in.data(), n);

        for (int i = 0; i < n; ++i) {
            h_out[i] = omega_ * diagonal_.data()[i] * h_in[i];
        }

        out.copy_from(h_out.data(), n);
    }

    T omega() const { return omega_; }

private:
    memory::Buffer<T> diagonal_;
    T omega_;
};

template<typename T>
class ILUPreconditioner : public Preconditioner<T> {
public:
    ILUPreconditioner() = default;

    void setup(const SparseMatrix<T>& A) override;

    void apply(const T* in, T* out) override;

    void apply(const memory::Buffer<T>& in, memory::Buffer<T>& out) override;

private:
    memory::Buffer<T> L_vals_;
    memory::Buffer<T> U_vals_;
    memory::Buffer<int> L_row_offsets_;
    memory::Buffer<int> L_col_indices_;
    memory::Buffer<int> U_row_offsets_;
    memory::Buffer<int> U_col_indices_;
    int n_ = 0;
};

template<typename T>
void ILUPreconditioner<T>::setup(const SparseMatrix<T>& A) {
    n_ = A.rows();
    (void)A;
}

template<typename T>
void ILUPreconditioner<T>::apply(const T* in, T* out) {
    (void)in;
    (void)out;
}

template<typename T>
void ILUPreconditioner<T>::apply(const memory::Buffer<T>& in, memory::Buffer<T>& out) {
    (void)in;
    (void)out;
}

}
