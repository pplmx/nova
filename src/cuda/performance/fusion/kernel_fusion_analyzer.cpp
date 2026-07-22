#include <cuda/performance/fusion/kernel_fusion_analyzer.h>
#include <cuda/performance/fusion/fusion_patterns.h>

#include <algorithm>
#include <cmath>

namespace cuda::performance::fusion {

std::vector<FusionPattern> KernelFusionAnalyzer::get_known_patterns() {
    // Delegate to the single source of truth in fusion_patterns.cpp so the two
    // definitions can't drift apart. Caller gets a copy (FusionPattern is a
    // small POD-ish struct) so KernelFusionAnalyzer can still own its
    // patterns_ member safely.
    return FusionPatterns::all();
}

KernelFusionAnalyzer::KernelFusionAnalyzer() {
    patterns_ = get_known_patterns();
}

void KernelFusionAnalyzer::add_operation(const Operation& op) {
    operations_.push_back(op);
}

void KernelFusionAnalyzer::add_operations(const std::vector<Operation>& ops) {
    operations_.insert(operations_.end(), ops.begin(), ops.end());
}

void KernelFusionAnalyzer::clear() {
    operations_.clear();
}

std::vector<FusionOpportunity> KernelFusionAnalyzer::detect_opportunities() const {
    return detect_opportunities(operations_);
}

std::vector<FusionOpportunity> KernelFusionAnalyzer::detect_opportunities(
    const std::vector<Operation>& ops) const {

    std::vector<FusionOpportunity> opportunities;

    for (size_t i = 0; i + 1 < ops.size(); ++i) {
        const auto& op1 = ops[i];
        const auto& op2 = ops[i + 1];

        auto applicable_patterns = get_applicable_patterns(op1);
        for (const auto& pattern : applicable_patterns) {
            if (matches_pattern(op1, op2, pattern)) {
                opportunities.emplace_back(op1, op2, pattern, i);
            }
        }
    }

    return opportunities;
}

std::vector<FusionPattern> KernelFusionAnalyzer::get_applicable_patterns(const Operation& op) const {
    std::vector<FusionPattern> applicable;

    for (const auto& pattern : patterns_) {
        if (!pattern.ops.empty() && pattern.ops[0] == op.type) {
            applicable.push_back(pattern);
        }
    }

    return applicable;
}

bool KernelFusionAnalyzer::matches_pattern(const Operation& op1, const Operation& op2,
                                            const FusionPattern& pattern) const {
    if (pattern.ops.size() < 2) return false;

    if (op1.type != pattern.ops[0]) return false;

    if (op1.type == OpType::Matmul || op1.type == OpType::Conv) {
        return op2.type == OpType::ElementWise;
    }

    if (pattern.ops.size() >= 2) {
        return op2.type == pattern.ops[1];
    }

    return false;
}

FusionOpportunity::FusionOpportunity(const Operation& first, const Operation& second,
                                      const FusionPattern& pattern, size_t index)
    : first_op_(first), second_op_(second), pattern_(pattern), location_index_(index) {}

uint64_t FusionOpportunity::combined_latency_ns() const {
    return first_op_.latency_ns + second_op_.latency_ns;
}

uint64_t FusionOpportunity::potential_latency_saved_ns() const {
    uint64_t combined = combined_latency_ns();
    double speedup = pattern_.estimated_speedup;
    if (speedup > 1.0) {
        return combined - static_cast<uint64_t>(combined / speedup);
    }
    return 0;
}

double FusionOpportunity::arithmetic_intensity() const {
    size_t total_flops = first_op_.flops + second_op_.flops;
    size_t total_bytes = first_op_.bytes_accessed + second_op_.bytes_accessed;
    if (total_bytes == 0) return 0.0;
    return static_cast<double>(total_flops) / static_cast<double>(total_bytes);
}

OpType parse_op_type(const std::string& name) {
    auto lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("matmul") != std::string::npos || lower.find("gemm") != std::string::npos) {
        return OpType::Matmul;
    }
    if (lower.find("conv") != std::string::npos) {
        return OpType::Conv;
    }
    if (lower.find("relu") != std::string::npos || lower.find("gelu") != std::string::npos ||
        lower.find("silu") != std::string::npos || lower.find("sigmoid") != std::string::npos) {
        return OpType::Activation;
    }
    if (lower.find("pool") != std::string::npos || lower.find("max") != std::string::npos ||
        lower.find("avg") != std::string::npos) {
        return OpType::Pooling;
    }
    if (lower.find("reduce") != std::string::npos || lower.find("sum") != std::string::npos ||
        lower.find("max") != std::string::npos) {
        return OpType::Reduction;
    }
    if (lower.find("add") != std::string::npos || lower.find("mul") != std::string::npos ||
        lower.find("bias") != std::string::npos) {
        return OpType::ElementWise;
    }
    if (lower.find("softmax") != std::string::npos) {
        return OpType::Softmax;
    }
    if (lower.find("layernorm") != std::string::npos || lower.find("norm") != std::string::npos) {
        return OpType::LayerNorm;
    }
    if (lower.find("dropout") != std::string::npos) {
        return OpType::Dropout;
    }
    return OpType::Unknown;
}

std::string to_string(OpType type) {
    switch (type) {
        case OpType::Matmul: return "Matmul";
        case OpType::Conv: return "Conv";
        case OpType::Activation: return "Activation";
        case OpType::Pooling: return "Pooling";
        case OpType::Reduction: return "Reduction";
        case OpType::ElementWise: return "ElementWise";
        case OpType::Softmax: return "Softmax";
        case OpType::LayerNorm: return "LayerNorm";
        case OpType::Dropout: return "Dropout";
        case OpType::Unknown: return "Unknown";
    }
    return "Unknown";
}

std::string to_string(const Operation& op) {
    return op.name + " (" + to_string(op.type) + ", " +
           std::to_string(op.latency_ns / 1000) + "us)";
}

}  // namespace cuda::performance::fusion
