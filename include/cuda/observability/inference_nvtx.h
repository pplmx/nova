#pragma once

#include "cuda/observability/nvtx_extensions.h"
#include <cstdint>
#include <string>

namespace cuda::observability {

    class InferenceNVTXDomain {
    public:
        static InferenceNVTXDomain& get() {
            static InferenceNVTXDomain instance;
            return instance;
        }

        void begin_prefill() {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
            nvtx3::mark("Prefill");
#endif
        }

        void end_prefill() {}

        void begin_decode() {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
            nvtx3::mark("Decode");
#endif
        }

        void end_decode() {}

        void begin_attention(const char* name = "Attention") {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
            nvtx3::mark(name);
#endif
        }

        void end_attention() {}

        void begin_scheduling() {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
            nvtx3::mark("Scheduling");
#endif
        }

        void end_scheduling() {}

        void record_batch_size(int size) {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
            nvtx3::mark(("BatchSize:" + std::to_string(size)).c_str());
#endif
        }

        void record_sequence_length(int length) {
#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED
            nvtx3::mark(("SeqLen:" + std::to_string(length)).c_str());
#endif
        }

    private:
        InferenceNVTXDomain() {}
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
        explicit ScopedAttention(const char* name = "Attention") { InferenceNVTXDomain::get().begin_attention(name); }
        ~ScopedAttention() { InferenceNVTXDomain::get().end_attention(); }
    };

    class ScopedScheduling {
    public:
        ScopedScheduling() { InferenceNVTXDomain::get().begin_scheduling(); }
        ~ScopedScheduling() { InferenceNVTXDomain::get().end_scheduling(); }
    };

}  // namespace cuda::observability
