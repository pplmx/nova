/**
 * @file move_semantics_audit_test.cpp
 * @brief Types that free resources must not be silently movable
 *
 * Two rounds ago SyncBatchNorm declared defaulted moves while its destructor freed nine
 * cudaMalloc'd pointers. A defaulted move copies those pointers without nulling the source,
 * so both destructors freed the same memory. That is now deleted.
 *
 * This file is the audit that was run after fixing it, kept so the property is checked rather
 * than remembered.
 *
 * The audit crosses every destructor that frees a CUDA resource against its class's move
 * declarations. A defaulted move alone is not enough to call it a defect: if the class holds
 * a std::mutex or std::shared_mutex, that member is not movable, and the defaulted move is
 * implicitly deleted. BlockManager is exactly that case — it looked like a candidate and is
 * not one. Getting this wrong cost a round: the fix was written, the static_assert that was
 * supposed to fail did not, and the reason was the shared_mutex member.
 *
 * The assertions below encode the outcome of the audit: the classes that own freed resources
 * are not movable, and the ones that looked like they might be are checked by the compiler
 * rather than by eye.
 */

#include <cuda/neural/sync_batch_norm.h>
#include <cuda/inference/block_manager.h>
#include <cuda/memory/unique_ptr.h>
#include <cuda/stream/stream.h>
#include <cuda/stream/event.h>
#include <cuda/production/priority_stream.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace move_semantics_audit_test {

// Each of these owns something its destructor releases. None may be moved or copied by
// default, because a defaulted move would leave two owners of one handle.
static_assert(!std::is_move_constructible_v<cuda::neural::SyncBatchNorm>,
              "SyncBatchNorm frees nine buffers in its destructor");
static_assert(!std::is_move_assignable_v<cuda::neural::SyncBatchNorm>);

// BlockManager holds a std::shared_mutex, which makes its defaulted move implicitly deleted.
// The assertion documents the conclusion rather than the declaration.
static_assert(!std::is_move_constructible_v<cuda::inference::BlockManager>,
              "BlockManager's shared_mutex member deletes the defaulted move");
static_assert(!std::is_move_assignable_v<cuda::inference::BlockManager>);

// These two are move-only by design: their moves null the source, so exactly one owner
// survives. The contrast with the deleted cases above is the point.
static_assert(std::is_move_constructible_v<cuda::memory::unique_ptr<float>>);
static_assert(!std::is_copy_constructible_v<cuda::memory::unique_ptr<float>>);
static_assert(std::is_move_constructible_v<cuda::stream::Stream>);
static_assert(!std::is_copy_constructible_v<cuda::stream::Stream>);
static_assert(std::is_move_constructible_v<cuda::stream::Event>);
static_assert(!std::is_copy_constructible_v<cuda::stream::Event>);

TEST(MoveSemanticsAuditTest, TheAuditHolds) {
    // The file's content is the static_asserts above; this case exists so the audit appears
    // in a test run and not only in a compile.
    SUCCEED();
}

}  // namespace move_semantics_audit_test
