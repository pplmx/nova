#include "cuda/sparse/cusparse_context.hpp"
#include "cuda/device/error.h"

namespace nova::sparse::detail {

CusparseContext::CusparseContext() {
    cusparseStatus_t status = cusparseCreate(&handle_);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuSPARSE handle: " +
                                  std::to_string(static_cast<int>(status)));
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
