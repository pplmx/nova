/**
 * @file nccl_enum_examples_test.cpp
 * @brief NCCL's op and type enums are C-style and must not be written with a scope
 *
 * Two headers showed the scoped form in their usage examples:
 *
 *     ncclRedOp_t::ncclSum
 *     ncclDataType_t::ncclFloat32
 *
 * Neither compiles. ncclRedOp_t and ncclDataType_t are typedefs in nccl.h, not classes or
 * namespaces, so `::` is a syntax error. A caller copying the example got a build failure with
 * nothing to suggest the example itself was wrong. The headers now show the bare names.
 *
 * The static_asserts below are the check: if either enum ever changed into a scoped
 * enumeration, these cases stop compiling and the examples' wording gets revisited. Compiling
 * this file is most of what it verifies.
 */

#include <cuda/nccl/nccl_all_reduce.h>
#include <cuda/nccl/nccl_broadcast.h>

#include <gtest/gtest.h>

namespace nccl_enum_examples_test {

    // The bare names are what the examples must use.
    static_assert(sizeof(ncclSum) > 0, "ncclSum is a value, not a scoped enumerator");
    static_assert(sizeof(ncclFloat32) > 0, "ncclFloat32 is a value");
    static_assert(sizeof(ncclMax) > 0, "ncclMax is a value");

    TEST(NcclEnumExamplesTest, TheOperationEnumsHaveTheValuesNcclDefines) {
        // Pinned so a locale-dependent or reordered enum would show up here rather than as a
        // silently wrong reduction in a caller.
        EXPECT_EQ(static_cast<int>(ncclSum), 0);
        EXPECT_EQ(static_cast<int>(ncclProd), 1);
        EXPECT_EQ(static_cast<int>(ncclMax), 2);
    }

    TEST(NcclEnumExamplesTest, TheDataTypeEnumsHaveTheValuesNcclDefines) {
        EXPECT_EQ(static_cast<int>(ncclInt8), 0);
        EXPECT_EQ(static_cast<int>(ncclFloat32), 7);
    }

}  // namespace nccl_enum_examples_test
