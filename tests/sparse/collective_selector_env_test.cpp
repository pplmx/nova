/**
 * @file collective_selector_env_test.cpp
 * @brief Contracts for NCCL algorithm selection and its environment bridge
 *
 * src/cuda/topology/topology_map.cpp picks a collective algorithm and can publish
 * it to the process environment. Two defects were found.
 *
 * - set_nccl_env wrote the selection into the wrong variable:
 *
 *       env << "NCCL_ALGO=" << algorithm_to_string(...);
 *       setenv("NCCL_TUNING", env.str().c_str(), 1);
 *
 *   NCCL reads the algorithm from NCCL_ALGO, so the choice never reached it.
 *   Measured: NCCL_ALGO was (unset) while NCCL_TUNING held
 *   "NCCL_ALGO=CollNet" — a function whose whole purpose is to configure NCCL had
 *   no effect on NCCL.
 *
 * - select_for_allgather returned Ring unconditionally, discarding both of its
 *   arguments, while select_for_allreduce and select_for_broadcast applied the
 *   same 32 MiB threshold and 8-rank rule. Measured with a 128 MiB message and 64
 *   workers: allgather said Ring while broadcast said Tree.
 */

#include <cuda/topology/topology_map.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace cuda::topology::test {

namespace {

/// Restores an environment variable to its previous state when the test ends, so
/// the mutations these tests make cannot leak into another case.
class ScopedEnv {
public:
    explicit ScopedEnv(const char* name) : name_(name) {
        if (const char* previous = std::getenv(name)) {
            saved_ = previous;
            had_previous_ = true;
        }
    }

    ~ScopedEnv() {
        if (had_previous_) {
            setenv(name_.c_str(), saved_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

    std::string value() const {
        const char* current = std::getenv(name_.c_str());
        return current ? current : std::string();
    }

private:
    std::string name_;
    std::string saved_;
    bool had_previous_ = false;
};

}  // namespace

// Regression: this wrote NCCL_TUNING and left NCCL_ALGO unset.
TEST(CollectiveSelectorEnvTest, SetNcclEnvPublishesTheAlgorithmToNcclAlgo) {
    ScopedEnv algo("NCCL_ALGO");
    ScopedEnv tuning("NCCL_TUNING");

    AlgorithmConfig config;
    config.algorithm = CollectiveAlgorithm::CollNet;

    CollectiveSelector::set_nccl_env(config);

    EXPECT_EQ(algo.value(), "CollNet")
        << "NCCL reads the algorithm from NCCL_ALGO";
    EXPECT_TRUE(tuning.value().empty() || tuning.value() != "NCCL_ALGO=CollNet")
        << "the algorithm must not be smuggled through NCCL_TUNING, got "
        << tuning.value();
}

TEST(CollectiveSelectorEnvTest, SetNcclEnvSpellsEachAlgorithm) {
    ScopedEnv algo("NCCL_ALGO");

    const struct {
        CollectiveAlgorithm algorithm;
        const char* expected;
    } cases[] = {
        {CollectiveAlgorithm::Ring, "Ring"},
        {CollectiveAlgorithm::Tree, "Tree"},
        {CollectiveAlgorithm::CollNet, "CollNet"},
        {CollectiveAlgorithm::Pipeline, "Pipeline"},
    };

    for (const auto& item : cases) {
        AlgorithmConfig config;
        config.algorithm = item.algorithm;
        CollectiveSelector::set_nccl_env(config);
        EXPECT_EQ(algo.value(), item.expected);
    }
}

TEST(CollectiveSelectorEnvTest, TuningFragmentIsAppendedNotReplaced) {
    ScopedEnv algo("NCCL_ALGO");

    AlgorithmConfig config;
    config.algorithm = CollectiveAlgorithm::Tree;
    config.nccl_tuning_env = "LL128";

    CollectiveSelector::set_nccl_env(config);

    const std::string published = algo.value();
    EXPECT_NE(published.find("Tree"), std::string::npos)
        << "the algorithm must survive: " << published;
    EXPECT_NE(published.find("LL128"), std::string::npos)
        << "the tuning fragment must survive: " << published;
}

// Regression: this ignored both arguments.
TEST(CollectiveSelectorEnvTest, AllgatherUsesTheSameThresholdAsBroadcast) {
    // A large message across many workers: both entry points should agree that a
    // tree beats a ring, because they share the same heuristic inputs.
    const auto allgather =
        CollectiveSelector::select_for_allgather(128ull * 1024 * 1024, 64);
    const auto broadcast =
        CollectiveSelector::select_for_broadcast(128ull * 1024 * 1024, 64);

    EXPECT_EQ(allgather.algorithm, CollectiveAlgorithm::Tree);
    EXPECT_EQ(allgather.algorithm, broadcast.algorithm)
        << "allgather and broadcast apply the same rule to the same inputs";
}

TEST(CollectiveSelectorEnvTest, AllgatherKeepsRingForSmallMessages) {
    const auto small = CollectiveSelector::select_for_allgather(1024, 64);
    EXPECT_EQ(small.algorithm, CollectiveAlgorithm::Ring)
        << "a ring is the right choice for a small message";
}

TEST(CollectiveSelectorEnvTest, AllgatherKeepsRingForFewWorkers) {
    // Enough bytes but too few participants to amortise a tree.
    const auto few =
        CollectiveSelector::select_for_allgather(128ull * 1024 * 1024, 4);
    EXPECT_EQ(few.algorithm, CollectiveAlgorithm::Ring);
}

TEST(CollectiveSelectorEnvTest, BroadcastAndAllreduceRejectSmallMessages) {
    EXPECT_EQ(CollectiveSelector::select_for_broadcast(1024, 64).algorithm,
              CollectiveAlgorithm::Ring);
    EXPECT_EQ(CollectiveSelector::select_for_allreduce(1024, false, "Ethernet")
                  .algorithm,
              CollectiveAlgorithm::Ring);
}

TEST(CollectiveSelectorEnvTest, AllreduceUpgradesForRdmaWithLargeMessages) {
    // Documented thresholds: 64 MiB selects CollNet over InfiniBand, 32 MiB
    // selects Tree, and below that a ring is used.
    EXPECT_EQ(CollectiveSelector::select_for_allreduce(128ull * 1024 * 1024, true,
                                                        "InfiniBand")
                  .algorithm,
              CollectiveAlgorithm::CollNet);
    EXPECT_EQ(CollectiveSelector::select_for_allreduce(48ull * 1024 * 1024, true,
                                                        "InfiniBand")
                  .algorithm,
              CollectiveAlgorithm::Tree);
    EXPECT_EQ(CollectiveSelector::select_for_allreduce(128ull * 1024 * 1024, false,
                                                        "Ethernet")
                  .algorithm,
              CollectiveAlgorithm::Tree)
        << "without RDMA the collnet path is unavailable, tree still applies";
}

TEST(CollectiveSelectorEnvTest, FromEnvReadsEachName) {
    ScopedEnv algo("NCCL_ALGO");

    const struct {
        const char* text;
        CollectiveAlgorithm expected;
    } cases[] = {
        {"Ring", CollectiveAlgorithm::Ring},
        {"Tree", CollectiveAlgorithm::Tree},
        {"CollNet", CollectiveAlgorithm::CollNet},
    };

    for (const auto& item : cases) {
        setenv("NCCL_ALGO", item.text, 1);
        const auto config = CollectiveSelector::from_env();
        EXPECT_EQ(config.algorithm, item.expected) << "for " << item.text;
    }
}

TEST(CollectiveSelectorEnvTest, SetThenFromEnvRoundTrips) {
    ScopedEnv algo("NCCL_ALGO");

    AlgorithmConfig original;
    original.algorithm = CollectiveAlgorithm::CollNet;
    CollectiveSelector::set_nccl_env(original);

    const auto restored = CollectiveSelector::from_env();
    EXPECT_EQ(restored.algorithm, original.algorithm)
        << "what set_nccl_env publishes must be what from_env reads back";
}

}  // namespace cuda::topology::test
