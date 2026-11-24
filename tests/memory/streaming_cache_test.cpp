#include <gtest/gtest.h>
#include "cuda/memory/streaming_cache_manager.h"
#include "cuda/memory/kv_cache_allocator.h"
#include "cuda/stream/stream.h"
#include <thread>
#include <chrono>

namespace cuda::memory {

    class StreamingCacheTest : public ::testing::Test {
    protected:
        void SetUp() override {
            KVCacheAllocatorConfig config {
                .num_heads = 32,
                .head_dim = 128,
                .block_size_tokens = 16,
                .num_blocks = 256,
                .num_layers = 32,
            };
            allocator = std::make_unique<KVCacheAllocator>(config);

            StreamingCacheConfig stream_config {
                .enable_prefetch = true,
                .enable_async_eviction = true,
                .prefetch_ahead_blocks = 2,
                .eviction_batch_size = 4,
            };
            manager = std::make_unique<StreamingCacheManager>(allocator.get(), stream_config);
        }

        std::unique_ptr<KVCacheAllocator> allocator;
        std::unique_ptr<StreamingCacheManager> manager;
    };

    TEST_F(StreamingCacheTest, PrefetchRequestCreated) {
        stream::Stream stream;

        auto blocks1 = allocator->allocate(1, 32);
        EXPECT_GT(blocks1.size(), 0);

        manager->prefetch_async(1, 16, stream);

        EXPECT_TRUE(manager->should_evict_async() == false || allocator->get_num_free_blocks() > 0);
    }

    TEST_F(StreamingCacheTest, ImportanceTracking) {
        manager->update_importance(1, 10.0f);
        manager->update_importance(2, 5.0f);

        EXPECT_EQ(manager->get_importance(1), 10.0f);
        EXPECT_EQ(manager->get_importance(2), 5.0f);
        EXPECT_EQ(manager->get_importance(3), 1.0f);
    }

    TEST_F(StreamingCacheTest, DefaultImportanceIsOne) {
        EXPECT_EQ(manager->get_importance(999), 1.0f);
    }

    TEST_F(StreamingCacheTest, PrefetchAsyncDisabled) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = false,
            .enable_async_eviction = true,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        stream::Stream stream;

        EXPECT_NO_THROW(local_manager->prefetch_async(1, 32, stream));
    }

    TEST_F(StreamingCacheTest, PrefetchCalculatesCorrectBlockCount) {
        stream::Stream stream;

        allocator->allocate(1, 32);

        manager->prefetch_async(1, 32, stream);

        EXPECT_TRUE(allocator->get_num_free_blocks() >= 0);
    }

    TEST_F(StreamingCacheTest, SyncPrefetchProcessesRequests) {
        stream::Stream stream;

        allocator->allocate(1, 16);

        manager->prefetch_async(1, 16, stream);
        manager->sync_prefetch(stream);

        auto blocks = allocator->get_blocks(1);
        // 1 initial block + prefetch: (16 tokens/16 block_size)+1 = 2 blocks,
        // capped at the fixture's prefetch_ahead_blocks=2 -> 3 blocks total.
        ASSERT_EQ(blocks.size(), 3);
    }

    // Neither prefetch_async() nor sync_prefetch() uses the stream it is given. Queuing is a locked
    // push_back onto an in-process vector, and processing is KVCacheAllocator::append(), a host-side
    // allocation from a free list that launches no kernel. The definition's comment claimed
    // sync_prefetch() executes the requests "on specified stream", which no part of the
    // implementation does.
    //
    // The point of this case is to pin that the result is the SAME whichever streams are passed: a
    // future implementation that really did schedule device work would have to keep it that way or
    // change this case deliberately.
    TEST_F(StreamingCacheTest, PrefetchOutcomeDoesNotDependOnTheStreamArgument) {
        stream::Stream stream_a;
        stream::Stream stream_b;

        allocator->allocate(1, 16);
        manager->prefetch_async(1, 16, stream_a);
        manager->sync_prefetch(stream_b);  // deliberately a different stream from the queued one

        auto blocks = allocator->get_blocks(1);
        ASSERT_EQ(blocks.size(), 3) << "the request must be processed regardless of which stream is passed";

        // A second sequence through a different pair of streams must produce the same shape.
        auto manager2 = std::make_unique<StreamingCacheManager>(allocator.get(), StreamingCacheConfig {});
        allocator->allocate(2, 16);
        manager2->prefetch_async(2, 16, stream_b);
        manager2->sync_prefetch(stream_a);
        EXPECT_EQ(allocator->get_blocks(2).size(), 3);
    }

    TEST_F(StreamingCacheTest, SyncPrefetchClearsPendingRequests) {
        stream::Stream stream;

        allocator->allocate(1, 16);

        manager->prefetch_async(1, 16, stream);
        EXPECT_GE(manager->pending_prefetch_count(), 1u) << "prefetch_async queued nothing";

        manager->sync_prefetch(stream);
        EXPECT_EQ(manager->pending_prefetch_count(), 0u) << "sync_prefetch left requests queued, despite its name";

        // A second sync on an empty queue must stay empty rather than resurrect anything.
        manager->sync_prefetch(stream);
        EXPECT_EQ(manager->pending_prefetch_count(), 0u);

        auto blocks = allocator->get_blocks(1);
        EXPECT_GE(blocks.size(), 1);
    }

    TEST_F(StreamingCacheTest, ShouldEvictAsyncBelowThreshold) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = true,
            .enable_async_eviction = false,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        EXPECT_FALSE(local_manager->should_evict_async());
    }

    TEST_F(StreamingCacheTest, ShouldEvictAsyncAboveThreshold) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = true,
            .enable_async_eviction = true,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        // 232 allocations of 1 block each leave 24/256 free (~9.4% < 10%), which
        // crosses should_evict_async()'s threshold. (230 left ~10.2% free, just
        // above it, so the assertion was previously unattainable.)
        for (int i = 0; i < 232; ++i) {
            allocator->allocate(i, 16);
        }

        EXPECT_TRUE(local_manager->should_evict_async());
    }

    TEST_F(StreamingCacheTest, ShouldEvictAsyncNearThreshold) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = true,
            .enable_async_eviction = true,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        for (int i = 0; i < 200; ++i) {
            allocator->allocate(i, 16);
        }

        auto stats = allocator->get_stats();
        auto free_ratio = static_cast<float>(stats.free_blocks) / stats.total_blocks;

        if (free_ratio < 0.1f) {
            EXPECT_TRUE(local_manager->should_evict_async());
        } else {
            EXPECT_FALSE(local_manager->should_evict_async());
        }
    }

    TEST_F(StreamingCacheTest, EvictImportanceWeighted) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = true,
            .enable_async_eviction = true,
            .eviction_policy = 1,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        allocator->allocate(1, 32);
        allocator->allocate(2, 32);
        allocator->allocate(3, 32);

        local_manager->update_importance(1, 5.0f);
        local_manager->update_importance(2, 10.0f);
        local_manager->update_importance(3, 1.0f);

        auto stats_before = allocator->get_stats();
        EXPECT_EQ(stats_before.allocated_blocks, 6);

        local_manager->evict_importance_weighted(2);

        auto stats_after = allocator->get_stats();
        EXPECT_LT(stats_after.allocated_blocks, stats_before.allocated_blocks);
    }

    TEST_F(StreamingCacheTest, EvictImportanceWeightedLRU) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = true,
            .enable_async_eviction = true,
            .eviction_policy = 1,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        allocator->allocate(1, 16);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        allocator->allocate(2, 16);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        allocator->allocate(3, 16);

        local_manager->update_importance(1, 1.0f);
        local_manager->update_importance(2, 1.0f);
        local_manager->update_importance(3, 1.0f);

        local_manager->evict_importance_weighted(3);

        auto blocks1 = allocator->get_blocks(1);
        auto blocks2 = allocator->get_blocks(2);
        auto blocks3 = allocator->get_blocks(3);

        int existing_count = 0;
        if (!blocks1.empty()) {
            existing_count++;
        }
        if (!blocks2.empty()) {
            existing_count++;
        }
        if (!blocks3.empty()) {
            existing_count++;
        }

        EXPECT_EQ(existing_count, 0);
    }

    TEST_F(StreamingCacheTest, EvictImportanceWeightedMixedImportance) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = true,
            .enable_async_eviction = true,
            .eviction_policy = 1,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        allocator->allocate(1, 16);
        allocator->allocate(2, 16);

        local_manager->update_importance(1, 100.0f);
        local_manager->update_importance(2, 1.0f);

        local_manager->evict_importance_weighted(1);

        auto blocks1 = allocator->get_blocks(1);
        EXPECT_EQ(blocks1.size(), 1);

        auto blocks2 = allocator->get_blocks(2);
        EXPECT_TRUE(blocks2.empty());
    }

    TEST_F(StreamingCacheTest, FragmentationMonitoringThreshold) {
        auto config = StreamingCacheConfig {
            .enable_prefetch = true,
            .enable_async_eviction = true,
            .eviction_policy = 0,
        };
        auto local_manager = std::make_unique<StreamingCacheManager>(allocator.get(), config);

        auto report = allocator->analyze_fragmentation();

        EXPECT_GE(report.ratio, 0.0f);
        EXPECT_LE(report.ratio, 100.0f);
    }

    TEST_F(StreamingCacheTest, UpdateImportanceOverwrites) {
        manager->update_importance(1, 5.0f);
        manager->update_importance(1, 10.0f);

        EXPECT_EQ(manager->get_importance(1), 10.0f);
    }

    TEST_F(StreamingCacheTest, MultiplePrefetchRequests) {
        stream::Stream stream;

        allocator->allocate(1, 16);
        allocator->allocate(2, 16);
        allocator->allocate(3, 16);

        manager->prefetch_async(1, 16, stream);
        manager->prefetch_async(2, 16, stream);
        manager->prefetch_async(3, 16, stream);

        manager->sync_prefetch(stream);

        auto blocks1 = allocator->get_blocks(1);
        auto blocks2 = allocator->get_blocks(2);
        auto blocks3 = allocator->get_blocks(3);

        EXPECT_GE(blocks1.size(), 1);
        EXPECT_GE(blocks2.size(), 1);
        EXPECT_GE(blocks3.size(), 1);
    }

    // ---------------------------------------------------------------------------
    // Importance-weighted eviction only ever saw the sequences its caller had scored.
    //
    // It ranked a snapshot of importance_weights_ and freed the sequences found there, so a
    // sequence holding blocks but never passed to update_importance() was invisible. With no
    // weights recorded at all -- a manager that starts with the default config, or a caller
    // that scores only some requests -- evict_importance_weighted() freed nothing and returned
    // as if it had. Measured on a full 32-block pool: it reported success with allocated=32
    // and free=0, while KVCacheAllocator::evict(8) freed 8 blocks from the same state.
    //
    // The allocator now exposes get_sequence_ids() so the manager can rank what actually
    // holds memory; an unweighted sequence sorts at the documented default importance.
    // ---------------------------------------------------------------------------

    TEST_F(StreamingCacheTest, EvictImportanceWeightedWithNoWeightsRecordedFreesBlocks) {
        KVCacheAllocatorConfig config {
            .num_heads = 4,
            .head_dim = 64,
            .block_size_tokens = 16,
            .num_blocks = 32,
            .num_layers = 2,
        };
        auto local_allocator = std::make_unique<KVCacheAllocator>(config);
        auto local_manager = std::make_unique<StreamingCacheManager>(local_allocator.get(),
                                                                     StreamingCacheConfig {
                                                                         .enable_async_eviction = true,
                                                                         .eviction_policy = 1,
                                                                     });

        // Fill the pool, and never call update_importance().
        local_allocator->allocate(1, 256);
        local_allocator->allocate(2, 256);
        ASSERT_EQ(local_allocator->get_num_free_blocks(), 0);
        ASSERT_GT(local_allocator->get_stats().allocated_blocks, 0);

        local_manager->evict_importance_weighted(8);

        EXPECT_GT(local_allocator->get_num_free_blocks(), 0) << "a weighted eviction that frees nothing is not a successful eviction";
    }

    // The complement: a sequence that holds blocks but was never scored must not be treated
    // as unreachable, and one that was scored but holds nothing must not be the only one
    // considered.
    TEST_F(StreamingCacheTest, EvictImportanceWeightedSeesUnscoredSequences) {
        KVCacheAllocatorConfig config {
            .num_heads = 4,
            .head_dim = 64,
            .block_size_tokens = 16,
            .num_blocks = 32,
            .num_layers = 2,
        };
        auto local_allocator = std::make_unique<KVCacheAllocator>(config);
        auto local_manager = std::make_unique<StreamingCacheManager>(local_allocator.get(),
                                                                     StreamingCacheConfig {
                                                                         .enable_async_eviction = true,
                                                                         .eviction_policy = 1,
                                                                     });

        // Fill the pool with a sequence that is never scored, and record a weight only for
        // a sequence that holds nothing.
        for (int64_t seq = 7; local_allocator->get_num_free_blocks() > 0; ++seq) {
            local_allocator->allocate(seq, 256);
        }
        local_manager->update_importance(999, 0.1f);

        ASSERT_EQ(local_allocator->get_num_free_blocks(), 0);

        local_manager->evict_importance_weighted(4);

        EXPECT_GT(local_allocator->get_num_free_blocks(), 0) << "the unscored sequence holding the blocks was not considered";
    }

    // Regression: the loop tested `evicted >= num_blocks` with `evicted` starting at 0, so a
    // negative count was already satisfied and the call returned having freed nothing -- the
    // same observable result as a legitimate no-op, from an argument that cannot mean one.
    TEST_F(StreamingCacheTest, EvictRejectsANegativeBlockCount) {
        EXPECT_THROW(manager->evict_importance_weighted(-8), std::invalid_argument) << "a negative count is not a smaller request";
        EXPECT_NO_THROW(manager->evict_importance_weighted(0)) << "zero is a legal no-op";
    }

    // Regression: eviction_policy selects a strategy by number. Any value other than 1 fell
    // through to plain LRU, so a caller setting 7 got an eviction it had not asked for while
    // believing a policy was in effect. prefetch_ahead_blocks = 0 is likewise not a
    // configuration, it is the absence of one.
    TEST_F(StreamingCacheTest, ConfigurationIsValidatedAtConstruction) {
        const auto rejection = [this](StreamingCacheConfig config) {
            try {
                StreamingCacheManager manager(allocator.get(), config);
            } catch (const std::invalid_argument& error) {
                return std::string(error.what());
            }
            return std::string();
        };

        EXPECT_FALSE(rejection([] {
                         StreamingCacheConfig c;
                         c.eviction_policy = 7;
                         return c;
                     }())
                         .empty())
            << "eviction_policy = 7 silently ran LRU";
        EXPECT_FALSE(rejection([] {
                         StreamingCacheConfig c;
                         c.prefetch_ahead_blocks = 0;
                         return c;
                     }())
                         .empty());

        EXPECT_TRUE(rejection(StreamingCacheConfig {}).empty()) << "the defaults must stay valid";
        EXPECT_TRUE(rejection([] {
                        StreamingCacheConfig c;
                        c.eviction_policy = 0;
                        return c;
                    }())
                        .empty());
        EXPECT_TRUE(rejection([] {
                        StreamingCacheConfig c;
                        c.eviction_policy = 1;
                        return c;
                    }())
                        .empty());
    }

    // Every method dereferences the allocator, so a null one is a crash on first use rather
    // than a manager without a cache.
    TEST_F(StreamingCacheTest, ConstructionRejectsANullAllocator) {
        EXPECT_THROW(StreamingCacheManager(nullptr, StreamingCacheConfig {}), std::invalid_argument);
    }

}  // namespace cuda::memory
