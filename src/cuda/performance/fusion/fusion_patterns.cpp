#include <cuda/performance/fusion/fusion_patterns.h>

#include <algorithm>
#include <unordered_map>

namespace cuda::performance::fusion {

    static std::vector<FusionPattern>& get_pattern_cache() {
        static std::vector<FusionPattern> patterns = {{"matmul_bias_act_relu", {OpType::Matmul, OpType::ElementWise, OpType::Activation}, "Matmul + bias + ReLU fusion", 1.5},
                                                      {"matmul_bias_act_gelu", {OpType::Matmul, OpType::ElementWise, OpType::Activation}, "Matmul + bias + GELU fusion", 1.4},
                                                      {"matmul_bias_act_silu", {OpType::Matmul, OpType::ElementWise, OpType::Activation}, "Matmul + bias + SiLU fusion", 1.4},
                                                      {"conv_bias_act_relu", {OpType::Conv, OpType::ElementWise, OpType::Activation}, "Convolution + bias + ReLU fusion", 1.6},
                                                      {"conv_bias_act_gelu", {OpType::Conv, OpType::ElementWise, OpType::Activation}, "Convolution + bias + GELU fusion", 1.5},
                                                      {"relu_pool", {OpType::Activation, OpType::Pooling}, "ReLU + pooling fusion", 1.2},
                                                      {"elementwise_chain", {OpType::ElementWise, OpType::ElementWise, OpType::ElementWise}, "Element-wise operation chain", 1.3},
                                                      {"reduction_norm", {OpType::Reduction, OpType::LayerNorm}, "Reduction + normalization fusion", 1.1},
                                                      {"softmax_dropout", {OpType::Softmax, OpType::Dropout}, "Softmax + dropout fusion", 1.2},
                                                      {"matmul_bias", {OpType::Matmul, OpType::ElementWise}, "Matmul + bias fusion", 1.2}};
        return patterns;
    }

    const std::vector<FusionPattern>& FusionPatterns::all() {
        return get_pattern_cache();
    }

    const FusionPattern* FusionPatterns::find_by_name(const std::string& name) {
        const auto& patterns = get_pattern_cache();
        for (const auto& p : patterns) {
            if (p.name == name) {
                return &p;
            }
        }
        return nullptr;
    }

    std::vector<const FusionPattern*> FusionPatterns::find_by_op_type(OpType type) {
        const auto& patterns = get_pattern_cache();
        std::vector<const FusionPattern*> result;
        for (const auto& p : patterns) {
            if (!p.ops.empty() && p.ops[0] == type) {
                result.push_back(&p);
            }
        }
        return result;
    }

    std::vector<const FusionPattern*> FusionPatterns::find_profitable(double min_speedup) {
        const auto& patterns = get_pattern_cache();
        std::vector<const FusionPattern*> result;
        for (const auto& p : patterns) {
            if (p.estimated_speedup >= min_speedup) {
                result.push_back(&p);
            }
        }
        return result;
    }

}  // namespace cuda::performance::fusion
