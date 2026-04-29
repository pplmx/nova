#pragma once

#include "cuda/observability/nvtx_extensions.h"
#include <cstdint>

namespace cuda::observability {

class InferenceNVTXDomain {
public:
    static InferenceNVTXDomain& get() {
        static InferenceNVTXDomain instance;
        return instance;
    }

    void begin_prefill() {
        nvtx3::scoped_range range("Prefill");
    }

    void end_prefill() {}

    void begin_decode() {
        nvtx3::scoped_range range("Decode");
    }

    void end_decode() {}

    void begin_attention(const char* name = "Attention") {
        nvtx3::scoped_range range(name);
    }

    void end_attention() {}

    void begin_scheduling() {
        nvtx3::scoped_range range("Scheduling");
    }

    void end_scheduling() {}

    void record_batch_size(int size) {
        nvtx3::mark(("BatchSize:" + std::to_string(size)).c_str());
    }

    void record_sequence_length(int length) {
        nvtx3::mark(("SeqLen:" + std::to_string(length)).c_str());
    }

private:
    InferenceNVTXDomain() {
        nvtx3::domain_handle_t domain = nvtxDomainCreateA("nova_inference");
        (void)domain;
    }
};

class ScopedPrefill {
public:
    ScopedPrefill() { InferenceNVTXDomain::get().begin_prefill(); }
    ~ScopedPrefill() { InferenceNVTXDomain::get().end_prefill(); }
};

class ScopedDecode {
public:
    ScopedDecode() { InferenceNVTXDomain::get().begin_decode(); }
    ~ScopedDecode() { InferenceNVTXDomain::get().end_decode(); }
};

class ScopedAttention {
public:
    explicit ScopedAttention(const char* name = "Attention") {
        InferenceNVTXDomain::get().begin_attention(name);
    }
    ~ScopedAttention() { InferenceNVTXDomain::get().end_attention(); }
};

class ScopedScheduling {
public:
    ScopedScheduling() { InferenceNVTXDomain::get().begin_scheduling(); }
    ~ScopedScheduling() { InferenceNVTXDomain::get().end_scheduling(); }
};

}  // namespace cuda::observability
