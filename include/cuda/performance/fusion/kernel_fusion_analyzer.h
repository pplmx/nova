#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace cuda::performance::fusion {

    enum class OpType { Matmul, Conv, Activation, Pooling, Reduction, ElementWise, Softmax, LayerNorm, Dropout, Unknown };

    enum class ActivationType { None, ReLU, GELU, SiLU, Sigmoid, Tanh, LeakyReLU };

    struct Operation {
        std::string name;
        OpType type;
        uint64_t latency_ns;
        size_t flops;
        size_t bytes_accessed;
        int grid_x, grid_y, grid_z;
        int block_x, block_y, block_z;
        float occupancy;
        bool is_memory_bound;
    };

    struct FusionPattern {
        std::string name;
        std::vector<OpType> ops;
        std::string description;
        double estimated_speedup;
    };

    class FusionOpportunity {
    public:
        FusionOpportunity() = default;
        FusionOpportunity(const Operation& first, const Operation& second, const FusionPattern& pattern, size_t index);

        [[nodiscard]] const Operation& first_op() const { return first_op_; }
        [[nodiscard]] const Operation& second_op() const { return second_op_; }
        [[nodiscard]] const FusionPattern& pattern() const { return pattern_; }
        [[nodiscard]] size_t location_index() const { return location_index_; }

        [[nodiscard]] uint64_t combined_latency_ns() const;
        [[nodiscard]] uint64_t potential_latency_saved_ns() const;
        [[nodiscard]] double arithmetic_intensity() const;

    private:
        Operation first_op_;
        Operation second_op_;
        FusionPattern pattern_;
        size_t location_index_{0};
    };

    class KernelFusionAnalyzer {
    public:
        KernelFusionAnalyzer();

        void add_operation(const Operation& op);
        void add_operations(const std::vector<Operation>& ops);
        void clear();

        [[nodiscard]] std::vector<FusionOpportunity> detect_opportunities() const;
        [[nodiscard]] std::vector<FusionOpportunity> detect_opportunities(const std::vector<Operation>& ops) const;

        static std::vector<FusionPattern> get_known_patterns();

    private:
        [[nodiscard]] bool matches_pattern(const Operation& op1, const Operation& op2, const FusionPattern& pattern) const;
        [[nodiscard]] std::vector<FusionPattern> get_applicable_patterns(const Operation& op) const;

        std::vector<Operation> operations_;
        std::vector<FusionPattern> patterns_;
    };

    OpType parse_op_type(const std::string& name);
    std::string to_string(OpType type);
    std::string to_string(const Operation& op);

}  // namespace cuda::performance::fusion
