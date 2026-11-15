#include "cuda/sparse/cusparse_context.hpp"
#include "cuda/device/error.h"

namespace nova::sparse::detail {

namespace {

/// cuSPARSE exposes no name lookup; report the enumerator rather than a bare
/// integer so a failure says what went wrong.
const char* cusparse_status_name(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
        case CUSPARSE_STATUS_NOT_SUPPORTED: return "CUSPARSE_STATUS_NOT_SUPPORTED";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
        default: return "CUSPARSE_STATUS_UNKNOWN";
    }
}

}  // namespace

CusparseContext::CusparseContext() {
    cusparseStatus_t status = cusparseCreate(&handle_);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("Failed to create cuSPARSE handle: ") +
            cusparse_status_name(status) + " (" +
            std::to_string(static_cast<int>(status)) + ")");
    }
}

CusparseContext::~CusparseContext() {
    if (handle_) {
        cusparseDestroy(handle_);
        handle_ = nullptr;
    }
}

CusparseContext& CusparseContext::get() {
    // Heap-allocated singleton that is intentionally never destroyed.
    //
    // A function-local static's destructor runs at process exit and calls
    // cusparseDestroy, which touches the CUDA device context during runtime
    // teardown - the same exit-crash class that SIGSEGV'd inside libcuda for
    // NcclContext (Round 12) and MeshStreams (Round 13) singletons.
    // The CUDA driver reclaims device state when the process exits, so orderly
    // teardown at exit is unnecessary.
    static CusparseContext* instance = new CusparseContext();
    return *instance;
}

void CusparseContext::set_stream(cudaStream_t stream) {
    stream_ = stream;
    cusparseSetStream(handle_, stream);
}

}  // namespace nova::sparse::detail
