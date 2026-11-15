/**
 * @file nccl_types_test.cpp
 * @brief Contracts for the NCCL type mappings
 *
 * include/cuda/nccl/nccl_types.h maps CUDA data types and reduction operations
 * onto NCCL. It had no tests, and two defects lived in it:
 *
 * - Every stub constant used when NCCL is compiled out had the wrong value.
 *   ncclSystemError was 1 (real 2), ncclInProgress was 2 (real 7), the data
 *   types were numbered in an entirely different order, and ncclMax/ncclMin were
 *   SWAPPED — so a non-NCCL build computed a minimum where the caller asked for
 *   a maximum. ncclTimeout was invented; no released nccl.h defines it.
 *
 * - CUDA_R_32U had no case in to_nccl_dtype and fell through to the
 *   ncclFloat32 default, silently reinterpreting a uint32 collective as float.
 *
 * The assertions below compare against the values in the real nccl.h, so they
 * hold in both configurations: with NCCL the symbols come from the header, and
 * without it they come from the stubs that must match it.
 */

#include <cuda/nccl/nccl_types.h>

#include <gtest/gtest.h>

#include <string>

namespace cuda::nccl::test {

// ------------------------------------------------ constants match nccl.h ----
//
// These are the values every released nccl.h uses. Reading them from the stub
// block when NCCL is absent is exactly what makes a non-NCCL build behave like
// the real one.

TEST(NcclConstantsTest, ResultCodesMatchTheRealHeader) {
    EXPECT_EQ(ncclSuccess, 0);
    EXPECT_EQ(ncclUnhandledCudaError, 1);
    EXPECT_EQ(ncclSystemError, 2);
    EXPECT_EQ(ncclInternalError, 3);
    EXPECT_EQ(ncclInvalidArgument, 4);
    EXPECT_EQ(ncclInvalidUsage, 5);
    EXPECT_EQ(ncclRemoteError, 6);
    EXPECT_EQ(ncclInProgress, 7);
}

TEST(NcclConstantsTest, ResultCodesAreDistinct) {
    const int codes[] = {ncclSuccess,      ncclUnhandledCudaError, ncclSystemError,
                         ncclInternalError, ncclInvalidArgument, ncclInvalidUsage,
                         ncclRemoteError, ncclInProgress};
    constexpr size_t count = sizeof(codes) / sizeof(codes[0]);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            EXPECT_NE(codes[i], codes[j])
                << "result codes " << i << " and " << j << " collide";
        }
    }
}

TEST(NcclConstantsTest, DataTypesMatchTheRealHeader) {
    // The order is part of the ABI: these numbers go into the collective call.
    EXPECT_EQ(ncclInt8, 0);
    EXPECT_EQ(ncclUint8, 1);
    EXPECT_EQ(ncclInt32, 2);
    EXPECT_EQ(ncclUint32, 3);
    EXPECT_EQ(ncclInt64, 4);
    EXPECT_EQ(ncclUint64, 5);
    EXPECT_EQ(ncclFloat16, 6);
    EXPECT_EQ(ncclFloat32, 7);
    EXPECT_EQ(ncclFloat64, 8);
    EXPECT_EQ(ncclBfloat16, 9);
}

TEST(NcclConstantsTest, DataTypesAreDistinct) {
    const int types[] = {ncclInt8,   ncclUint8,   ncclInt32,   ncclUint32,  ncclInt64,
                         ncclUint64, ncclFloat16, ncclFloat32, ncclFloat64, ncclBfloat16};
    constexpr size_t count = sizeof(types) / sizeof(types[0]);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            EXPECT_NE(types[i], types[j])
                << "data types " << i << " and " << j << " collide";
        }
    }
}

// Regression: Max and Min were the other way round, so the wrong extreme was
// computed whenever NCCL was not compiled in.
TEST(NcclConstantsTest, ReductionOpsMatchTheRealHeader) {
    EXPECT_EQ(ncclSum, 0);
    EXPECT_EQ(ncclProd, 1);
    EXPECT_EQ(ncclMax, 2);
    EXPECT_EQ(ncclMin, 3);

    EXPECT_NE(ncclMax, ncclMin) << "max and min must not share a value";
    EXPECT_LT(ncclMax, ncclMin)
        << "the real header orders ncclMax before ncclMin; if this flips, the "
           "stub block regressed to the swapped version";
}

// ------------------------------------------------------ to_nccl_dtype ----

TEST(ToNcclDtypeTest, MapsEveryStandardCudaType) {
    EXPECT_EQ(to_nccl_dtype(CUDA_R_8I), ncclInt8);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_8U), ncclUint8);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_16F), ncclFloat16);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_16BF), ncclBfloat16);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_32F), ncclFloat32);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_64F), ncclFloat64);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_32I), ncclInt32);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_64I), ncclInt64);
    EXPECT_EQ(to_nccl_dtype(CUDA_R_64U), ncclUint64);
}

// Regression: CUDA_R_32U had no case and defaulted to ncclFloat32, so a uint32
// collective reinterpreted its data as float.
TEST(ToNcclDtypeTest, UnsignedInt32IsNotSilentlyAFloat) {
    const ncclDataType_t mapped = to_nccl_dtype(CUDA_R_32U);

    EXPECT_EQ(mapped, ncclUint32);
    EXPECT_NE(mapped, ncclFloat32)
        << "a uint32 buffer must not be sent as float data";
    EXPECT_NE(mapped, ncclInt32) << "signedness would be lost";
}

TEST(ToNcclDtypeTest, MappedTypesRoundTripThroughTheirNames) {
    // The name lookup is what error messages and logs report, so a mapping that
    // resolves to a type the namer cannot identify is a diagnostic hole.
    const cudaDataType_t inputs[] = {CUDA_R_8I,  CUDA_R_8U,  CUDA_R_16F,  CUDA_R_16BF,
                                     CUDA_R_32F, CUDA_R_64F, CUDA_R_32I,  CUDA_R_32U,
                                     CUDA_R_64I, CUDA_R_64U};
    for (const cudaDataType_t input : inputs) {
        const std::string name = dtype_name(to_nccl_dtype(input));
        EXPECT_NE(name, "unknown")
            << "cudaDataType_t " << static_cast<int>(input)
            << " maps to a type dtype_name cannot identify";
    }
}

// ---------------------------------------------------------------- to_nccl_op ----

TEST(ToNcclOpTest, MapsEveryReductionOperation) {
    using ::cuda::distributed::ReductionOp;

    EXPECT_EQ(to_nccl_op(ReductionOp::Sum), ncclSum);
    EXPECT_EQ(to_nccl_op(ReductionOp::Product), ncclProd);
    EXPECT_EQ(to_nccl_op(ReductionOp::Max), ncclMax);
    EXPECT_EQ(to_nccl_op(ReductionOp::Min), ncclMin);
}

// Regression: with the swapped stubs, Max resolved to ncclMin's value.
TEST(ToNcclOpTest, MaxAndMinResolveToDifferentOperations) {
    using ::cuda::distributed::ReductionOp;

    const ncclRedOp_t max_op = to_nccl_op(ReductionOp::Max);
    const ncclRedOp_t min_op = to_nccl_op(ReductionOp::Min);

    EXPECT_NE(max_op, min_op) << "max and min must not be the same reduction";
    EXPECT_EQ(max_op, ncclMax);
    EXPECT_EQ(min_op, ncclMin);
}

// ---------------------------------------------------------------- dtype_name ----

TEST(DtypeNameTest, EveryMappedTypeHasAName) {
    const ncclDataType_t types[] = {ncclInt8,   ncclUint8,   ncclInt32,   ncclUint32,
                                    ncclInt64,  ncclUint64,  ncclFloat16, ncclFloat32,
                                    ncclFloat64, ncclBfloat16};
    for (const ncclDataType_t type : types) {
        EXPECT_NE(std::string(dtype_name(type)), "unknown")
            << "type " << static_cast<int>(type) << " has no name";
    }
}

TEST(DtypeNameTest, NamesAreDistinctAndMeaningful) {
    EXPECT_STREQ(dtype_name(ncclInt8), "int8");
    EXPECT_STREQ(dtype_name(ncclUint8), "uint8");
    EXPECT_STREQ(dtype_name(ncclInt32), "int32");
    EXPECT_STREQ(dtype_name(ncclUint32), "uint32");
    EXPECT_STREQ(dtype_name(ncclInt64), "int64");
    EXPECT_STREQ(dtype_name(ncclUint64), "uint64");
    EXPECT_STREQ(dtype_name(ncclFloat16), "float16");
    EXPECT_STREQ(dtype_name(ncclFloat32), "float32");
    EXPECT_STREQ(dtype_name(ncclFloat64), "float64");
    EXPECT_STREQ(dtype_name(ncclBfloat16), "bfloat16");
}

}  // namespace cuda::nccl::test
