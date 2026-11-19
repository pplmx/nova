/**
 * @file nvtx_sparse.hpp
 * @brief NVTX profiling markers for sparse operations
 * @defgroup nvtx_sparse NVTX Markers
 * @ingroup sparse
 *
 * Provides NVTX markers for profiling sparse operations.
 * Only active when NOVA_NVTX is defined.
 *
 * @see observability/nvtx.hpp For general NVTX support
 */

#ifndef NOVA_CUDA_SPARSE_NVTX_SPARSE_HPP
#define NOVA_CUDA_SPARSE_NVTX_SPARSE_HPP

#ifdef NOVA_NVTX

#include <nvtx3/nvtx3.hpp>

namespace nova {
    namespace sparse {
        namespace nvtx {

            /**
             * @brief Get NVTX domain for sparse operations
             * @return NVTX domain handle
             */
            inline nvtxDomainHandle_t get_sparse_domain() {
                static nvtxDomainHandle_t domain = nvtxDomainCreateA("nova_sparse");
                return domain;
            }

            /**
             * @brief RAII NVTX range wrapper
             * @class ScopedRange
             * @ingroup nvtx_sparse
             */
            struct ScopedRange {
                /** @brief Start NVTX range */
                ScopedRange(const char* name) { nvtxRangePushA(name); }
                /** @brief End NVTX range */
                ~ScopedRange() { nvtxRangePop(); }
            };

#define NOVA_NVTX_SCOPED_RANGE(name) nova::sparse::nvtx::ScopedRange nova_nvtx_range_(name)
#define NOVA_NVTX_MARKER(name) nvtxMarkA(name)

        }  // namespace nvtx
    }  // namespace sparse
}  // namespace nova

#else

namespace nova {
    namespace sparse {
        namespace nvtx {
            inline void* get_sparse_domain() {
                return nullptr;
            }
        }  // namespace nvtx
    }  // namespace sparse
}  // namespace nova

#define NOVA_NVTX_SCOPED_RANGE(name) ((void)0)
#define NOVA_NVTX_MARKER(name) ((void)0)

#endif

#endif
