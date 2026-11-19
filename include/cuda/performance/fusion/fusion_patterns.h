#pragma once

#include <cuda/performance/fusion/kernel_fusion_analyzer.h>

#include <vector>

namespace cuda::performance::fusion {

    struct FusionPatterns {
        static constexpr int PATTERN_COUNT = 10;

        static const std::vector<FusionPattern>& all();
        static const FusionPattern* find_by_name(const std::string& name);
        static std::vector<const FusionPattern*> find_by_op_type(OpType type);
        static std::vector<const FusionPattern*> find_profitable(double min_speedup);
    };

    struct FusionPatternRegistry {
        using Factory = FusionPattern (*)();
        static void register_pattern(const std::string& name, Factory factory);
        static const FusionPattern* get(const std::string& name);
        static std::vector<std::string> list_names();
    };

}  // namespace cuda::performance::fusion
