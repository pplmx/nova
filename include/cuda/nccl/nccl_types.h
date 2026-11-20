#pragma once

/**
 * @file nccl_types.h
 * @brief NCCL type mappings and data type conversions
 *
 * Provides conversion functions between CUDA data types and NCCL data types.
 * All NCCL-specific code is conditionally compiled based on NOVA_NCCL_ENABLED.
 *
 * @example
 * @code
 * // Convert CUDA dtype to NCCL dtype for collective operations
 * cudaDataType cuda_dtype = CUDA_R_32F;
 * ncclDataType_t nccl_dtype = cuda::nccl::to_nccl_dtype(cuda_dtype);
 *
 * // Convert reduction operation
 * ReductionOp op = ReductionOp::Sum;
 * ncclRedOp_t nccl_op = cuda::nccl::to_nccl_op(op);
 * @endcode
 */

#include "cuda/distributed/common.h"

#include <cuda_runtime.h>

#include <cstddef>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#else
// Stub definitions for non-NCCL builds
struct ncclComm_v2 {
    int dummy;
};
using ncclComm_t = struct ncclComm_v2*;
using ncclRedOp_t = int;
using ncclDataType_t = int;
using ncclResult_t = int;

// Stub values for NCCL result codes.
//
// These MUST equal the real ncclResult_t values. They did not: SystemError was
// 1 (real 2), InProgress was 2 (real 7), and ncclTimeout does not exist in any
// released nccl.h. Code that maps a status back to a severity in a non-NCCL
// build then classified the wrong failure.
constexpr int ncclSuccess = 0;
constexpr int ncclUnhandledCudaError = 1;
constexpr int ncclSystemError = 2;
constexpr int ncclInternalError = 3;
constexpr int ncclInvalidArgument = 4;
constexpr int ncclInvalidUsage = 5;
constexpr int ncclRemoteError = 6;
constexpr int ncclInProgress = 7;

// Stub values for common types, matching ncclDataType_t.
constexpr int ncclInt8 = 0;
constexpr int ncclUint8 = 1;
constexpr int ncclInt32 = 2;
constexpr int ncclUint32 = 3;
constexpr int ncclInt64 = 4;
constexpr int ncclUint64 = 5;
constexpr int ncclFloat16 = 6;
constexpr int ncclFloat32 = 7;
constexpr int ncclFloat64 = 8;
constexpr int ncclBfloat16 = 9;

// Stub values for reduction ops, matching ncclRedOp_t. Max and Min were
// previously defined the other way round, so a non-NCCL build computed a
// minimum where the caller asked for a maximum.
constexpr int ncclSum = 0;
constexpr int ncclProd = 1;
constexpr int ncclMax = 2;
constexpr int ncclMin = 3;

// Stub inline no-op functions for non-NCCL builds
inline ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) {
    return ncclResult_t(0);
}
inline ncclResult_t ncclBroadcast(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
    return ncclResult_t(0);
}
inline ncclResult_t ncclBarrier(ncclComm_t, cudaStream_t) {
    return ncclResult_t(0);
}
inline ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) {
    return ncclResult_t(0);
}
inline ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) {
    return ncclResult_t(0);
}
inline int ncclGroupStart() {
    return 0;
}
inline int ncclGroupEnd() {
    return 0;
}
#endif

namespace cuda::nccl {

    /**
     * @brief Map CUDA data type to NCCL data type
     *
     * @param cuda_dtype CUDA data type (cudaDataType_t)
     * @return Corresponding NCCL data type (ncclDataType_t)
     *
     * @note Maps all standard CUDA types to their NCCL equivalents.
     *       Unknown types default to ncclFloat32.
     *
     * @example
     * @code
     * ncclDataType_t dtype = to_nccl_dtype(CUDA_R_16F);  // Returns ncclFloat16
     * @endcode
     */
    [[nodiscard]]
    inline constexpr ncclDataType_t to_nccl_dtype(cudaDataType_t cuda_dtype) {
        // No #if here. The stub block above defines every ncclDataType_t constant
        // with the value the real nccl.h uses, so one table serves both
        // configurations and the mapping can no longer disagree with itself. The
        // branches that used to sit here returned a constant — 0 for the type,
        // which is ncclInt8, for every input — so a non-NCCL build mapped
        // CUDA_R_32F to int8 and reported every type as "unknown".
        switch (cuda_dtype) {
            case CUDA_R_8I:
                return ncclInt8;
            case CUDA_R_8U:
                return ncclUint8;
            case CUDA_R_16F:
                return ncclFloat16;
            case CUDA_R_16BF:
                return ncclBfloat16;
            case CUDA_R_32F:
                return ncclFloat32;
            case CUDA_R_64F:
                return ncclFloat64;
            case CUDA_R_32I:
                return ncclInt32;
            // CUDA_R_32U had no case and fell through to ncclFloat32, so a
            // uint32 collective silently reinterpreted its data as float.
            case CUDA_R_32U:
                return ncclUint32;
            case CUDA_R_64I:
                return ncclInt64;
            case CUDA_R_64U:
                return ncclUint64;
            default:
                return ncclFloat32;
        }
    }

    /**
     * @brief Map reduction operation to NCCL reduction operation
     *
     * @param op Nova reduction operation (ReductionOp enum)
     * @return Corresponding NCCL reduction operation (ncclRedOp_t)
     *
     * @note Maps all standard reduction operations. Unknown operations
     *       default to ncclSum.
     *
     * @example
     * @code
     * ncclRedOp_t op = to_nccl_op(ReductionOp::Sum);   // Returns ncclSum
     * ncclRedOp_t op = to_nccl_op(ReductionOp::Min);   // Returns ncclMin
     * ncclRedOp_t op = to_nccl_op(ReductionOp::Max);   // Returns ncclMax
     * ncclRedOp_t op = to_nccl_op(ReductionOp::Prod);  // Returns ncclProd
     * @endcode
     */
    [[nodiscard]]
    inline constexpr ncclRedOp_t to_nccl_op(::cuda::distributed::ReductionOp op) {
        // No #if, for the same reason as to_nccl_dtype: the stub constants match
        // nccl.h. The old non-NCCL branch returned 0 (ncclSum) for every input, so
        // a Max reduction on such a build computed a sum.
        switch (op) {
            case ::cuda::distributed::ReductionOp::Sum:
                return ncclSum;
            case ::cuda::distributed::ReductionOp::Product:
                return ncclProd;
            case ::cuda::distributed::ReductionOp::Min:
                return ncclMin;
            case ::cuda::distributed::ReductionOp::Max:
                return ncclMax;
            default:
                return ncclSum;
        }
    }

    /**
     * @brief Get human-readable name for NCCL data type
     *
     * @param dtype NCCL data type
     * @return String name of the type
     */
    [[nodiscard]]
    inline const char* dtype_name(ncclDataType_t dtype) {
        // No #if: the stub constants match nccl.h, and the old non-NCCL branch
        // returned "unknown" for every type, which made every diagnostic on such a
        // build name no type at all.
        switch (dtype) {
            case ncclInt8:
                return "int8";
            case ncclUint8:
                return "uint8";
            case ncclInt32:
                return "int32";
            case ncclUint32:
                return "uint32";
            case ncclInt64:
                return "int64";
            case ncclUint64:
                return "uint64";
            case ncclFloat16:
                return "float16";
            case ncclFloat32:
                return "float32";
            case ncclFloat64:
                return "float64";
            case ncclBfloat16:
                return "bfloat16";
            default:
                return "unknown";
        }
    }

}  // namespace cuda::nccl
