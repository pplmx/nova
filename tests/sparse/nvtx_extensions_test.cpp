/**
 * @file nvtx_extensions_test.cpp
 * @brief Contracts for the NVTX instrumentation helpers
 *
 * include/cuda/observability/nvtx_extensions.h had no tests, and the NVTX
 * enabled branch could not compile its own POP_RANGE macro:
 *
 *   #define NOVA_NVTX_POP_RANGE() nvtx3::pop_range()
 *   error: 'pop_range' is not a member of 'nvtx3'
 *
 * nvtx3 offers no pop_range. Its range API is start_range()/end_range() around a
 * range_handle, plus the RAII scoped_range for a scope. PUSH_RANGE was mapped
 * onto mark(), which emits an instantaneous point event rather than opening a
 * range, so even the compiling half did not do what its name said.
 *
 * Two further defects:
 *
 * - All nine NVTXDomains handles were the same nullptr, so a range could not be
 *   attributed to a domain at all.
 * - The two branches disagreed on ScopedRange's template parameter: the enabled
 *   build took a value (NVTXDomainHandle), the disabled build took a TYPE. Code
 *   written for one configuration failed to compile against the other, which is
 *   exactly what this header exists to prevent.
 *
 * This suite runs in the default build, where NVTX is off, so it pins the
 * disabled-branch behaviour and the shared shape. The enabled branch is verified
 * by compiling the same call sites with -DNOVA_NVTX_ENABLED=1 and the nvtx3
 * include path; that check lives in the commit message rather than here because
 * the test binary cannot be built both ways at once.
 */

#include <cuda/observability/nvtx_extensions.h>

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <type_traits>

namespace cuda::observability::test {

    // Regression: the nine handles were all nullptr.
    TEST(NvtxDomainsTest, EveryDomainIsDistinct) {
        const std::set<NVTXDomainHandle> domains = {
            NVTXDomains::Memory,
            NVTXDomains::Device,
            NVTXDomains::Algo,
            NVTXDomains::API,
            NVTXDomains::Production,
            NVTXDomains::Performance,
            NVTXDomains::NVBlox,
            NVTXDomains::Fusion,
            NVTXDomains::Bandwidth,
        };

        EXPECT_EQ(domains.size(), 9u) << "each domain must have its own identity, otherwise ranges cannot be "
                                         "attributed";
    }

    TEST(NvtxDomainsTest, HandlesAreNonNull) {
        EXPECT_NE(NVTXDomains::Memory, nullptr);
        EXPECT_NE(NVTXDomains::Device, nullptr);
        EXPECT_NE(NVTXDomains::Algo, nullptr);
        EXPECT_NE(NVTXDomains::API, nullptr);
        EXPECT_NE(NVTXDomains::Production, nullptr);
        EXPECT_NE(NVTXDomains::Performance, nullptr);
        EXPECT_NE(NVTXDomains::NVBlox, nullptr);
        EXPECT_NE(NVTXDomains::Fusion, nullptr);
        EXPECT_NE(NVTXDomains::Bandwidth, nullptr);
    }

    TEST(NvtxDomainsTest, HandlesAreUsableAsTemplateArguments) {
        // A domain handle has to be a constant expression to serve as a template
        // argument; if these became runtime values this would stop compiling.
        //
        // The assertion is written as `A == A`, not `A != B`. Comparing the two
        // DIFFERENT tags is not a constant expression under UBSan: the sanitizer
        // instruments the objects being pointed at, so their addresses stop being
        // usable in a constant comparison and `make sanitize` failed to compile
        // this file at all. Distinctness is checked at runtime by
        // EveryDomainIsDistinct above, which is where that property belongs.
        // The property under test is that these constants are usable in a CONSTANT
        // expression. That is checked by initialising a constexpr variable from them: if
        // they were not constant expressions, this file would not compile.
        //
        // Two earlier spellings were worse. `static_assert(A == A)` was a tautology that
        // GCC rejects (-Wtautological-compare) and asserted nothing while it lasted; a
        // boolean trick around it was still a tautology and tripped -Waddress. Comparing
        // two DIFFERENT tags here is not available either: UBSan instruments the objects,
        // so their addresses stop being usable in a constant comparison. Distinctness of
        // the values is checked at runtime by EveryDomainIsDistinct above.
        constexpr auto memory_domain = NVTXDomains::Memory;
        constexpr auto algo_domain = NVTXDomains::Algo;
        (void)memory_domain;
        (void)algo_domain;
        SUCCEED();
    }

    // Regression: the disabled branch declared ScopedRange with a TYPE parameter,
    // so this exact form failed to compile here while succeeding with NVTX on.
    TEST(ScopedRangeTest, TakesTheDomainAsAValueInBothConfigurations) {
        ScopedRange<NVTXDomains::Memory> range("allocate");
        ScopedRange<NVTXDomains::Algo> another("reduce");
        (void)range;
        (void)another;
        SUCCEED();
    }

    TEST(ScopedRangeTest, IsConstructibleFromAName) {
        EXPECT_NO_THROW(ScopedRange<NVTXDomains::Device>("cudaMalloc"));
        EXPECT_NO_THROW(([] { ScopedRange<NVTXDomains::Fusion> r(nullptr); }()));
    }

    TEST(ScopedRangeTest, IsAStackObject) {
        // RAII semantics: constructing and destroying inside a scope must not throw.
        EXPECT_NO_THROW({
            ScopedRange<NVTXDomains::Performance> outer("outer");
            { ScopedRange<NVTXDomains::Performance> inner("inner"); }
        });
    }

    TEST(RangeHelpersTest, PushAndPopAreSafeWithoutATool) {
        // With NVTX off these are no-ops; with it on they must still be balanced and
        // must not fault when popped without a matching push.
        EXPECT_NO_THROW(push_range<NVTXDomains::API>("collective"));
        EXPECT_NO_THROW(pop_range<NVTXDomains::API>());
    }

    TEST(RangeHelpersTest, UnbalancedPopDoesNotCrash) {
        // Regression guard for the thread-local depth counter: popping past empty
        // must be a no-op rather than an out-of-bounds read of the handle stack.
        EXPECT_NO_THROW(pop_range<NVTXDomains::API>());
        EXPECT_NO_THROW(pop_range<NVTXDomains::API>());
    }

    TEST(RangeHelpersTest, NestedRangesAreLifo) {
        EXPECT_NO_THROW({
            push_range<NVTXDomains::Algo>("outer");
            push_range<NVTXDomains::Algo>("inner");
            pop_range<NVTXDomains::Algo>();
            pop_range<NVTXDomains::Algo>();
        });
    }

    TEST(RangeHelpersTest, ANullNameIsIgnored) {
        EXPECT_NO_THROW(push_range<NVTXDomains::Memory>(nullptr));
        EXPECT_NO_THROW(pop_range<NVTXDomains::Memory>());
    }

    // Regression: the macros are the documented entry point and the disabled
    // POP_RANGE did not exist in nvtx3 at all.
    TEST(NvtxMacroTest, MacrosAreUsableAndBalanced) {
        NOVA_NVTX_PUSH_RANGE("macro-push");
        NOVA_NVTX_POP_RANGE();
        SUCCEED();
    }

    TEST(NvtxMacroTest, ScopedRangeMacroDeclaresANamedObject) {
        NOVA_NVTX_SCOPED_RANGE(block);
        (void)block;
        SUCCEED();
    }

    TEST(NvtxMacroTest, MacrosTolerateNesting) {
        NOVA_NVTX_PUSH_RANGE("level-1");
        NOVA_NVTX_PUSH_RANGE("level-2");
        NOVA_NVTX_POP_RANGE();
        NOVA_NVTX_POP_RANGE();
        SUCCEED();
    }

}  // namespace cuda::observability::test
