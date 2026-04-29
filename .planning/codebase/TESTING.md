# Testing Patterns

**Analysis Date:** 2026-04-30

## Test Framework

**Unit Testing:** GoogleTest (gtest v1.17.0) + GoogleMock (gmock)
- Fetched via CMake FetchContent
- Located in `tests/` directory

**Property-Based Testing:** Custom framework (`tests/property/property_test.hpp`)
- Uses `std::mt19937_64` for random generation
- Seeded for reproducibility

**Fuzz Testing:** libFuzzer (when using Clang)
- Enable with `cmake -DNOVA_BUILD_FUZZ_TESTS=ON`

**Benchmarking:** Google Benchmark (v1.9.1)
- Separate benchmark suite in `benchmark/` directory

## Test File Organization

### Location Pattern

```
tests/
├── CMakeLists.txt           # Test executable definitions
├── CMakeFuzz.txt            # Fuzz test configuration
├── CMakeProperty.txt        # Property test configuration
├── *.test.cpp / *.test.cu   # Unit tests (co-located with functionality)
├── unit/                    # Unit test organization (if separate)
├── property/                # Property-based tests
│   ├── property_test.hpp    # Test framework header
│   ├── mathematical_tests.cpp
│   ├── algorithmic_tests.cpp
│   └── numerical_tests.cpp
├── fuzz/                    # Fuzz test corpus
│   ├── memory_pool_fuzz.cpp
│   ├── algorithm_fuzz.cpp
│   └── matmul_fuzz.cpp
├── neural/
├── inference/
└── production/
```

### Test Naming

| Type | Pattern | Example |
|------|---------|---------|
| Unit | `<feature>_test.cpp` or `<feature>_test.cu` | `reduce_test.cu`, `memory_pool_test.cpp` |
| Property | `*_tests.cpp` | `numerical_tests.cpp`, `algorithmic_tests.cpp` |
| Fuzz | `<component>_fuzz.cpp` | `memory_pool_fuzz.cpp` |
| Integration | `integration_test.cpp` | `inference/integration_test.cpp` |
| Edge cases | `*_edge_test.cpp` | `flash_attention_edge_test.cu` |

## Test Structure

### GoogleTest Style

**Test fixture class:**
```cpp
class BufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();  // CUDA device reset
    }
    // Shared fixtures
    size_t size_ = 1024;
    std::vector<int> h_input_;
    cuda::memory::Buffer<int> d_input_;
};

TEST_F(BufferTest, ConstructionWithSize) {
    cuda::memory::Buffer<int> buffer(100);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 100);
}
```

**Naming convention:**
- Class: `<Feature>Test`
- Test: `TEST_F(FeatureTest, <Description>)`
- Description uses PascalCase: `SumBasic`, `ConstructionWithSize`, `NonPowerOfTwo`

### Test Categories

#### Unit Tests

**Scope:** Individual functions/classes in isolation
**Location:** `tests/*.cpp`, `tests/*.cu`
**Configuration:** Linked against `GTest::gtest_main`, `cuda_impl`, `cuda_api`

```cpp
// tests/reduce_test.cu
TEST_F(ReduceTest, SumBasic) {
    for (int i = 1; i <= static_cast<int>(size_); ++i) h_input_[i-1] = i;
    d_input_.copy_from(h_input_.data(), size_);

    int result = cuda::algo::reduce_sum(d_input_.data(), size_);
    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;

    EXPECT_EQ(result, expected);
}
```

#### Edge Case Tests

**Naming:** `*_edge_test.cpp` or `*_edge_test.cu`
**Scope:** Boundary conditions, error paths, unusual inputs

```cpp
TEST_F(ReduceTest, EmptyInput) {
    int result = cuda::algo::reduce_sum(d_input_.data(), 0);
    EXPECT_EQ(result, 0);
}

TEST_F(ReduceTest, NonPowerOfTwo) {
    h_input_.resize(1000);
    for (int i = 0; i < 1000; ++i) h_input_[i] = i + 1;
    d_input_ = cuda::memory::Buffer<int>(1000);
    d_input_.copy_from(h_input_.data(), 1000);
    int result = cuda::algo::reduce_sum(d_input_.data(), 1000);
    EXPECT_EQ(result, 1000 * 1001 / 2);
}
```

#### Property-Based Tests

**Framework:** Custom in `tests/property/property_test.hpp`

```cpp
// tests/property/numerical_tests.cpp
bool FloatPrecisionConsistency(RandomGenerator& gen) {
    auto a = gen.UniformFloat<double>(-1000.0, 1000.0);
    auto b = gen.UniformFloat<double>(-1000.0, 1000.0);

    double sum = a + b;
    double diff = a - b;
    double prod = a * b;
    double quot = a / b;

    return std::isfinite(sum) && std::isfinite(diff) &&
           std::isfinite(prod) && std::isfinite(quot);
}

int main(int argc, char** argv) {
    uint64_t seed = 0;
    if (argc > 1) {
        seed = std::stoull(argv[1]);
    }

    auto result = CheckProperty("Float Precision Consistency",
                                 FloatPrecisionConsistency, 100, seed);
    // ...
}
```

**Test executable targets:**
- `property_mathematical` - Mathematical invariants
- `property_algorithmic` - Algorithmic correctness
- `property_numerical` - Numerical stability

#### Fuzz Tests

**Framework:** libFuzzer
**Requires:** Clang compiler

```cpp
// tests/fuzz/memory_pool_fuzz.cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Parse input, exercise code under test
    // ...
    return 0;
}
```

**Corpus location:** `tests/fuzz/corpus/<target>/`
**Targets:**
- `memory_pool_fuzz` - Memory pool operations
- `algorithm_fuzz` - Algorithm inputs
- `matmul_fuzz` - Matrix multiplication inputs

#### Integration Tests

**Location:** Subdirectories with `integration_test.cpp`
**Scope:** Multiple components working together

```cpp
// tests/inference/integration_test.cpp
TEST(IntegrationTest, EndToEndPipeline) {
    // Test full pipeline from input to output
}
```

#### Production/Stress Tests

**Location:** `tests/production/`
**Scope:** Long-running, memory-intensive, observability

```cpp
// tests/production/stress_test.cpp
// tests/production/performance_test.cpp
// tests/production/observability_test.cpp
```

## Assertion Patterns

### GoogleTest Assertions

```cpp
// Equality
EXPECT_EQ(actual, expected);
ASSERT_EQ(actual, expected);  // Fatal

// NULL checks
EXPECT_NE(ptr, nullptr);
EXPECT_EQ(ptr, nullptr);

// Exception testing
EXPECT_THROW({ code; }, ExpectedExceptionType);

// Truth tests
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Float comparison (with tolerance)
EXPECT_FLOAT_EQ(actual, expected);
EXPECT_NEAR(actual, expected, 0.001);

// Death tests (for error handling)
EXPECT_DEATH({ code; }, "error pattern");
```

### CUDA-Specific Patterns

```cpp
// For GTEST_FAIL with custom message
if (err == cudaSuccess) {
    GTEST_FAIL() << "Memory should have been freed";
}

// Using ASSERT for fatal failures
ASSERT_NE(buffer.data(), nullptr);

// CUDA_CHECK is used for production code, not tests
// Tests verify behavior rather than catching exceptions
```

## Test Fixtures and Setup

### Device Reset

```cpp
namespace {
void reset() {
    cudaDeviceReset();
}
}

class BufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset();
    }
};
```

### Buffer Setup Pattern

```cpp
class ReduceTest : public ::testing::Test {
protected:
    size_t size_ = 1024;
    std::vector<int> h_input_;
    cuda::memory::Buffer<int> d_input_;

    void SetUp() override {
        h_input_.resize(size_);
        d_input_ = cuda::memory::Buffer<int>(size_);
    }

    void TearDown() override {
        d_input_.release();
    }
};
```

## Mocking

**Framework:** GoogleMock (GMock)

```cpp
#include <gmock/gmock.h>

class MockDeviceContext : public DeviceContextInterface {
public:
    MOCK_METHOD(cudaError_t, allocate, (void** ptr, size_t size), (override));
    MOCK_METHOD(cudaError_t, free, (void* ptr), (override));
};
```

## Benchmarking Patterns

**Framework:** Google Benchmark

```cpp
// benchmark/benchmark_kernels.cu
static void BM_MemoryH2D(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_data(n, 1.0f);
    float* d_data = allocate_device(n);

    for (auto _ : state) {
        NOVA_NVTX_SCOPED_RANGE("H2D_transfer");
        cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    state.SetBytesProcessed(int64_t(bytes * state.iterations()));
    state.SetItemsProcessed(int64_t(n * state.iterations()));

    free_device(d_data);
}

BENCHMARK(BM_MemoryH2D)
    ->RangeMultiplier(2)
    ->Ranges({{1 << 10, 1 << 24}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
```

## Run Commands

### Build and Test

```bash
# Configure
cmake -G Ninja -B build

# Build
cmake --build build --parallel

# Run all tests
ctest --test-dir build --output-on-failure -j16

# Run specific test
./build/bin/nova-tests --gtest_filter="ReduceTest.SumBasic"
```

### Property Tests

```bash
# Build
cmake --build build --parallel --target property_mathematical property_algorithmic property_numerical

# Run with seed
./build/bin/property_mathematical [seed]
./build/bin/property_algorithmic [seed]
./build/bin/property_numerical [seed]
```

### Fuzz Tests

```bash
# Configure with fuzz enabled
cmake -G Ninja -B build -DNOVA_BUILD_FUZZ_TESTS=ON

# Build targets
cmake --build build --parallel --target memory_pool_fuzz algorithm_fuzz matmul_fuzz

# Run fuzzers
./build/bin/memory_pool_fuzz tests/fuzz/corpus/memory_pool -max_total_time=60
```

### Coverage

```bash
# Configure with coverage
cmake -G Ninja -B build-coverage -DNOVA_COVERAGE=ON -DNOVA_COVERAGE_MIN=80

# Build
cmake --build build-coverage --parallel

# Run tests
ctest --test-dir build-coverage -j16

# Generate report
./scripts/coverage/generate_coverage.sh build-coverage
```

## CI/CD Integration

**Workflows:** `.github/workflows/`

### Testing Pipeline

```yaml
# .github/workflows/testing-quality.yml
unit-tests:
    runs-on: [self-hosted, gpu]
    steps:
        - cmake -G Ninja -B build
        - cmake --build build --parallel
        - ctest --test-dir build --output-on-failure -j16

property-tests:
    runs-on: [self-hosted, gpu]
    steps:
        - cmake --build build --parallel --target property_mathematical property_algorithmic property_numerical
        - ./build/bin/property_mathematical
        - ./build/bin/property_algorithmic
        - ./build/bin/property_numerical

fuzz-tests:
    runs-on: [self-hosted, gpu]
    steps:
        - cmake -G Ninja -B build -DNOVA_BUILD_FUZZ_TESTS=ON
        - cmake --build build --target memory_pool_fuzz algorithm_fuzz matmul_fuzz
        - timeout 300 ./build/bin/memory_pool_fuzz tests/fuzz/corpus/memory_pool -max_total_time=60

coverage:
    runs-on: [self-hosted, gpu]
    steps:
        - cmake -G Ninja -B build-coverage -DNOVA_COVERAGE=ON
        - cmake --build build-coverage
        - ctest --test-dir build-coverage
        - ./scripts/coverage/generate_coverage.sh build-coverage
```

## Coverage Requirements

**Minimum line coverage:** 80% (NOVA_COVERAGE_MIN)

**Enforcement:**
- Coverage job in CI runs on every push to main/PR
- Fails if line coverage drops below threshold
- Reports uploaded as artifacts

## Test Parallelism

**CPU parallelism:** Capped at `CTEST_PARALLEL_LEVEL` (default: 16, due to GPU memory constraints)

```bash
ctest -j16  # Run 16 tests in parallel
```

## Known Patterns

### Large Array Tests

```cpp
TEST_F(ReduceTest, LargeArray) {
    size_ = 1 << 20;  // 1M elements
    h_input_.resize(size_);
    d_input_ = cuda::memory::Buffer<int>(size_);
    // ...
}
```

### Stress Tests

```cpp
TEST_F(MemoryPoolStressTest, ManySmallAllocations) {
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        ptrs.push_back(pool.allocate(100));
        ASSERT_NE(ptrs.back(), nullptr);
    }
    // Deallocate
}
```

### Move Semantics Tests

```cpp
TEST_F(BufferTest, MoveConstruction) {
    cuda::memory::Buffer<int> buffer1(10);
    auto* data = buffer1.data();
    cuda::memory::Buffer<int> buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.data(), data);
    EXPECT_EQ(buffer1.data(), nullptr);  // Source is now empty
}
```

---

*Testing analysis: 2026-04-30*
