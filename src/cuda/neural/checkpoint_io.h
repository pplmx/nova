/**
 * @file checkpoint_io.h
 * @brief Internal deterministic binary IO for trainer checkpoints (v2.32)
 *
 * Private to the neural training sources (not installed): the NSCK-v1 stream
 * format shared by MicroTrainer::save_state/load_state and
 * TransformerTrainer::save_state/load_state. Layout is little-endian IEEE-754
 * floats and little-endian u32s written as native raw bytes. The library's
 * targets are little-endian CUDA hosts, so native writes coincide with the
 * documented layout; cross-endian file portability is not part of the format.
 *
 *   u32 magic  = 'N' 'S' 'C' 'K'  (0x4B43534E)
 *   u32 version = 3                (v2 adds the TP-degree field after kind; v3
 *                                   adds the writer's rank id after tp)
 *   u32 kind    = 1 (MicroTrainer) | 2 (TransformerTrainer)
 *   u32 tp      = TP degree of the trainer that wrote the file (load rejects
 *                 a mismatch, so only the same topology roundtrips and a
 *                 wrong-topology file fails up front, not by size luck)
 *   u32 rank    = rank id of the writer's shard (v3; load rejects a mismatch,
 *                 so a same-topology file written by another rank cannot
 *                 silently restore the wrong shard)
 *   ... trainer-specific dims, then per tensor a tagged record:
 *   u32 count; float[count]              (a weight tensor)
 *   and per optimizer a tagged moment pair:
 *   u32 count; float[count] m; float[count] v   (count 0 == fresh/zero moments)
 *
 * Every read validates: a wrong magic/version/kind is a runtime_error, a
 * count that doesn't match the receiving trainer's tensor is a runtime_error,
 * and a short read (truncated or corrupt stream) is a runtime_error — never
 * silent garbage.
 */

#pragma once

#include "cuda/device/error.h"
#include "cuda/nccl/nccl_context.h"
#include "cuda/neural/optimizers/optimizers.h"

#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::neural::training::checkpoint {

constexpr uint32_t kMagic = 0x4B43534Eu;  // 'N' 'S' 'C' 'K'
constexpr uint32_t kVersion = 3u;  // v2 adds TP degree; v3 adds the rank id
constexpr uint32_t kKindMicroTrainer = 1u;
constexpr uint32_t kKindTransformerTrainer = 2u;
constexpr uint32_t kMaxCount = 0xFFFFFFFFu;  // tensor/moment element cap (u32 field)

// The rank id that owns a checkpoint's shard set: the active thread's NCCL rank
// within ctx's group (the v2.20 verified active_rank convention — raw device
// indices are wrong for non-default NcclContextConfig groups) when the trainer
// is tensor-parallel, or 0 for the tp == 1 single-shard case (the only shard is
// shard 0; the context may be uninitialized there and rank_of_device throws on
// an uninitialized context).
inline uint32_t shard_rank(::cuda::nccl::NcclContext& ctx, int tp) {
    if (tp <= 1) {
        return 0u;
    }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return static_cast<uint32_t>(ctx.rank_of_device(device));
}

class Writer {
public:
    explicit Writer(std::ostream& out) : out_(out) {}

    void u32(uint32_t v) {
        out_.write(reinterpret_cast<const char*>(&v),
                   static_cast<std::streamsize>(sizeof(v)));
        if (!out_) {
            throw std::runtime_error("Nova checkpoint: write failed (stream error)");
        }
    }

    void floats(const float* data, size_t n) {
        if (n == 0) return;
        out_.write(reinterpret_cast<const char*>(data),
                   static_cast<std::streamsize>(n * sizeof(float)));
        if (!out_) {
            throw std::runtime_error("Nova checkpoint: write failed (stream error)");
        }
    }

    // A tagged tensor: element count, then raw floats.
    void tensor(const float* data, size_t n) {
        u32(checked(n));
        floats(data, n);
    }

private:
    static uint32_t checked(size_t n) {
        if (n > kMaxCount) {
            throw std::runtime_error("Nova checkpoint: tensor exceeds the count cap");
        }
        return static_cast<uint32_t>(n);
    }

    std::ostream& out_;
};

class Reader {
public:
    explicit Reader(std::istream& in) : in_(in) {}

    uint32_t u32() {
        uint32_t v = 0;
        read_exactly(&v, sizeof(v));
        return v;
    }

    void floats(float* data, size_t n) {
        if (n == 0) return;
        read_exactly(data, n * sizeof(float));
    }

    // A tagged tensor, validated to hold exactly n elements.
    void tensor(float* data, size_t n, const char* what) {
        const uint32_t count = u32();
        if (count != n) {
            throw std::runtime_error(
                std::string("Nova checkpoint: ") + what + " size mismatch "
                "(checkpoint " + std::to_string(count) + " vs trainer " +
                std::to_string(n) + ")");
        }
        floats(data, n);
    }

private:
    void read_exactly(void* dst, size_t bytes) {
        in_.read(static_cast<char*>(dst),
                 static_cast<std::streamsize>(bytes));
        if (static_cast<size_t>(in_.gcount()) != bytes) {
            throw std::runtime_error("Nova checkpoint: stream truncated");
        }
    }

    std::istream& in_;
};

// AdamW moment-pair record (count + m + v), shared by both trainers' save/load
// (TASK-057). A count of 0 means the saved optimizer was fresh (zero moments).
inline void write_adamw_moments(Writer& w,
                                const optimizers::AdamWOptimizer& opt) {
    const size_t cap = opt.momentum_capacity();
    if (cap > kMaxCount) {
        throw std::runtime_error("Nova checkpoint: moment count exceeds the cap");
    }
    w.u32(static_cast<uint32_t>(cap));
    if (cap == 0) return;
    std::vector<float> m(cap), v(cap);
    opt.copy_moments_to(m.data(), v.data(), cap, nullptr);
    w.floats(m.data(), cap);
    w.floats(v.data(), cap);
}

// A moment-pair read so far (no optimizer touched): used by the trainers' two
// phase load_state — every read validates before ANY state is applied, so a
// corrupt/truncated stream throws with the trainer untouched (no partial
// restore). m/v empty == the saved optimizer was fresh (count 0).
struct AdamWMomentRecord {
    std::vector<float> m;
    std::vector<float> v;
};

// Reads an AdamW moment-pair record, requiring the count to be either 0
// (fresh) or exactly the companion tensor's size.
inline AdamWMomentRecord read_adamw_moments(Reader& r, size_t tensor_n,
                                            const char* what) {
    const uint32_t count = r.u32();
    if (count != 0 && count != tensor_n) {
        throw std::runtime_error(
            std::string("Nova checkpoint: ") + what + " moment size mismatch "
            "(checkpoint " + std::to_string(count) + " vs tensor " +
            std::to_string(tensor_n) + ")");
    }
    AdamWMomentRecord rec;
    if (count == 0) return rec;
    rec.m.resize(count);
    rec.v.resize(count);
    r.floats(rec.m.data(), count);
    r.floats(rec.v.data(), count);
    return rec;
}

// Applies a record read by read_adamw_moments to an optimizer (call only after
// every record in the checkpoint validated).
inline void apply_adamw_moments(optimizers::AdamWOptimizer& opt,
                                const AdamWMomentRecord& rec) {
    if (rec.m.empty()) {
        opt.zero_momentum();
        return;
    }
    opt.copy_moments_from(rec.m.data(), rec.v.data(), rec.m.size(), nullptr);
}

}  // namespace cuda::neural::training::checkpoint
