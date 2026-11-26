#include <gtest/gtest.h>
#include <cuda/performance/fusion/kernel_fusion_analyzer.h>
#include <cuda/performance/fusion/fusion_profitability.h>
#include <cuda/performance/fusion/fusion_patterns.h>

namespace cuda::performance::fusion::test {

    class FusionAnalyzerTest : public ::testing::Test {
    protected:
        KernelFusionAnalyzer analyzer;
    };

    Operation make_op(const std::string& name, OpType type, uint64_t latency_ns = 1000) {
        return Operation {name, type, latency_ns, 1024, 64, 1, 1, 1, 128, 1, 1, 0.75f, false};
    }

    // FusionRecommendation divides by the pattern's estimated speedup without checking it:
    //
    //     after_latency_us_ = static_cast<uint64_t>(before_latency_us_ / speedup);
    //     latency_saved_us_ = before_latency_us_ - after_latency_us_;
    //
    // FusionPattern is a public aggregate and FusionOpportunity accepts any pattern, so a speedup
    // below 1 is reachable from caller-constructed input. Then after > before, and the two are
    // uint64_t: latency_saved_us_ underflows to about 1.8e19 and is summed into the exported
    // total_latency_saved_us. A speedup of exactly 0 makes the division infinite, and casting that
    // to uint64_t is undefined behaviour. kernel_fusion_analyzer.cpp guards the same comparison
    // (potential_latency_saved_ns() checks `speedup > 1.0`); this one did not.
    TEST_F(FusionAnalyzerTest, ANonPositiveSpeedupSavingsDoesNotUnderflow) {
        Operation first {"a", OpType::Matmul, 5000000, 1, 1, 1, 1, 1, 1, 1, 1, 1.0f, false};
        Operation second {"b", OpType::ElementWise, 5000000, 1, 1, 1, 1, 1, 1, 1, 1, 1.0f, false};

        FusionPattern slower {"slower", {OpType::Matmul, OpType::ElementWise}, "speedup below one", 0.8};
        FusionOpportunity slow_opp(first, second, slower, 0);
        const FusionRecommendation slow_rec(slow_opp, ConfidenceLevel::HIGH);

        // The latencies have to be large enough that before / speedup exceeds before AFTER the
        // truncation to uint64_t; a small pair (1000 ns each) divides back to the same integer and
        // the subtraction stays at zero, which would make this case vacuous.
        EXPECT_GT(slow_rec.before_latency_us(), 1000u) << "the case needs a latency large enough to expose the difference";

        // A fusion that makes things slower saves nothing; it must not wrap.
        EXPECT_LE(slow_rec.after_latency_us(), slow_rec.before_latency_us()) << "the fused latency must not exceed the unfused one for the subtraction to be meaningful";
        EXPECT_LT(slow_rec.latency_saved_us(), 1000000u) << "latency_saved_us underflowed";

        FusionPattern zero {"zero", {OpType::Matmul, OpType::ElementWise}, "speedup zero", 0.0};
        FusionOpportunity zero_opp(first, second, zero, 1);
        const FusionRecommendation zero_rec(zero_opp, ConfidenceLevel::HIGH);
        EXPECT_LT(zero_rec.latency_saved_us(), 1000000u) << "a zero speedup produced an underflowed saving";
    }

    TEST_F(FusionAnalyzerTest, DetectMatmulBiasPattern) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        EXPECT_GE(opportunities.size(), 1u);
    }

    TEST_F(FusionAnalyzerTest, DetectReluPoolPattern) {
        analyzer.add_operation(make_op("relu", OpType::Activation, 30));
        analyzer.add_operation(make_op("maxpool", OpType::Pooling, 40));

        auto opportunities = analyzer.detect_opportunities();
        EXPECT_GE(opportunities.size(), 1u);
    }

    TEST_F(FusionAnalyzerTest, NoOpportunitiesWithUnknownTypes) {
        analyzer.add_operation(make_op("unknown1", OpType::Unknown, 100));
        analyzer.add_operation(make_op("unknown2", OpType::Unknown, 100));

        auto opportunities = analyzer.detect_opportunities();
        EXPECT_EQ(opportunities.size(), 0u);
    }

    TEST_F(FusionAnalyzerTest, MultipleOpportunities) {
        analyzer.add_operation(make_op("matmul1", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias1", OpType::ElementWise, 50));
        analyzer.add_operation(make_op("matmul2", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias2", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        EXPECT_GE(opportunities.size(), 2u);
    }

    TEST_F(FusionAnalyzerTest, OpportunityLatencyCalculation) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        ASSERT_GT(opportunities.size(), 0u);

        EXPECT_EQ(opportunities[0].combined_latency_ns(), 150u);
        EXPECT_GT(opportunities[0].potential_latency_saved_ns(), 0u);
    }

    TEST_F(FusionAnalyzerTest, ClearOperations) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.clear();

        auto opportunities = analyzer.detect_opportunities();
        EXPECT_EQ(opportunities.size(), 0u);
    }

    TEST_F(FusionAnalyzerTest, ParseOpType) {
        EXPECT_EQ(parse_op_type("matmul_kernel"), OpType::Matmul);
        EXPECT_EQ(parse_op_type("GEMM"), OpType::Matmul);
        EXPECT_EQ(parse_op_type("conv2d"), OpType::Conv);
        EXPECT_EQ(parse_op_type("relu_forward"), OpType::Activation);
        EXPECT_EQ(parse_op_type("max_pool"), OpType::Pooling);
        EXPECT_EQ(parse_op_type("reduce_sum"), OpType::Reduction);
        EXPECT_EQ(parse_op_type("bias_add"), OpType::ElementWise);
    }

    TEST_F(FusionAnalyzerTest, ToStringOpType) {
        EXPECT_EQ(to_string(OpType::Matmul), "Matmul");
        EXPECT_EQ(to_string(OpType::Conv), "Conv");
        EXPECT_EQ(to_string(OpType::Activation), "Activation");
        EXPECT_EQ(to_string(OpType::Pooling), "Pooling");
    }

    class FusionProfitabilityTest : public ::testing::Test {
    protected:
        FusionProfitabilityModel model;
        KernelFusionAnalyzer analyzer;
    };

    TEST_F(FusionProfitabilityTest, DefaultConfig) {
        auto config = model.get_config();
        EXPECT_DOUBLE_EQ(config.launch_overhead_us, 100.0);
    }

    TEST_F(FusionProfitabilityTest, IsProfitable) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        ASSERT_GT(opportunities.size(), 0u);

        EXPECT_TRUE(model.is_profitable(opportunities[0]));
    }

    TEST_F(FusionProfitabilityTest, ProfitabilityScore) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        ASSERT_GT(opportunities.size(), 0u);

        double score = model.profitability_score(opportunities[0]);
        EXPECT_GE(score, 0.0);
        EXPECT_LE(score, 1.0);
    }

    TEST_F(FusionProfitabilityTest, LaunchOverheadSaved) {
        EXPECT_EQ(model.estimated_launch_overhead_saved_us(), 100u);
    }

    class FusionRecommendationTest : public ::testing::Test {
    protected:
        FusionRecommendationEngine engine;
        KernelFusionAnalyzer analyzer;
    };

    TEST_F(FusionRecommendationTest, GenerateRecommendations) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        auto recommendations = engine.generate_recommendations(opportunities);

        // `>= 0u` on a size_t asserts nothing. A matmul followed by a bias is the textbook
        // fusible pair, so the engine must produce at least one recommendation; asserting only
        // non-negativity let a generator that returned nothing pass this case.
        EXPECT_GE(recommendations.size(), 1u) << "a matmul plus bias must yield at least one fusion recommendation";
    }

    TEST_F(FusionRecommendationTest, RecommendationsSortedByScore) {
        analyzer.add_operation(make_op("matmul1", OpType::Matmul, 200));
        analyzer.add_operation(make_op("bias1", OpType::ElementWise, 50));
        analyzer.add_operation(make_op("relu", OpType::Activation, 30));
        analyzer.add_operation(make_op("pool", OpType::Pooling, 40));

        auto opportunities = analyzer.detect_opportunities();
        auto recommendations = engine.generate_recommendations(opportunities);

        for (size_t i = 1; i < recommendations.size(); ++i) {
            EXPECT_GE(recommendations[i - 1].profitability_score(), recommendations[i].profitability_score());
        }
    }

    TEST_F(FusionRecommendationTest, ConfidenceLevels) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        auto recommendations = engine.generate_recommendations(opportunities);

        for (const auto& rec : recommendations) {
            EXPECT_NE(to_string(rec.confidence()), "UNKNOWN");
        }
    }

    TEST_F(FusionRecommendationTest, JsonExport) {
        analyzer.add_operation(make_op("matmul", OpType::Matmul, 100));
        analyzer.add_operation(make_op("bias", OpType::ElementWise, 50));

        auto opportunities = analyzer.detect_opportunities();
        if (!opportunities.empty()) {
            FusionRecommendation rec(opportunities[0], ConfidenceLevel::HIGH);
            std::string json = rec.to_json();
            EXPECT_NE(json.find("pattern"), std::string::npos);
            EXPECT_NE(json.find("confidence"), std::string::npos);
        }
    }

    TEST_F(FusionRecommendationTest, CustomSuggestion) {
        engine.add_custom_suggestion("matmul_bias_act_relu", "Custom: Use FusedMatmulBiasAct kernel");
        EXPECT_EQ(engine.get_config().launch_overhead_us, 100.0);
    }

    class FusionPatternsTest : public ::testing::Test {};

    TEST_F(FusionPatternsTest, AllPatterns) {
        auto patterns = FusionPatterns::all();
        EXPECT_EQ(patterns.size(), 10u);
    }

    TEST_F(FusionPatternsTest, FindByName) {
        auto* pattern = FusionPatterns::find_by_name("matmul_bias_act_relu");
        EXPECT_NE(pattern, nullptr);
        EXPECT_EQ(pattern->name, "matmul_bias_act_relu");
    }

    TEST_F(FusionPatternsTest, FindByOpType) {
        auto patterns = FusionPatterns::find_by_op_type(OpType::Matmul);
        EXPECT_GE(patterns.size(), 3u);
    }

    TEST_F(FusionPatternsTest, FindProfitable) {
        auto patterns = FusionPatterns::find_profitable(1.3);
        EXPECT_GE(patterns.size(), 1u);
    }

    TEST_F(FusionPatternsTest, KnownPatternsMatch) {
        auto analyzer_patterns = KernelFusionAnalyzer::get_known_patterns();
        auto patterns = FusionPatterns::all();

        EXPECT_EQ(analyzer_patterns.size(), patterns.size());
    }

}  // namespace cuda::performance::fusion::test
