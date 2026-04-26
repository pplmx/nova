# Property-Based Testing

This directory contains property-based tests that verify mathematical invariants, algorithmic correctness, and numerical stability.

## Test Executables

| Executable | Description |
|------------|-------------|
| `property_mathematical` | Tests mathematical invariants (matmul identity, FFT inverse, transpose) |
| `property_algorithmic` | Tests algorithmic correctness (sort, reduce, scan) |
| `property_numerical` | Tests numerical stability across precision modes |

## Running Tests

```bash
# Build
cmake -B build && cmake --build build

# Run all property tests
./build/bin/property_mathematical
./build/bin/property_algorithmic
./build/bin/property_numerical

# Or run combined
make property_all
```

## Seed-Based Reproduction

Each test run generates a random seed for reproducibility:

```bash
# Run with specific seed
./build/bin/property_mathematical 12345
./build/bin/property_algorithmic 12345
./build/bin/property_numerical 12345
```

When a test fails, it outputs the seed used:
```
[FAIL] Test Name
  Seed: 9876543210 | Iterations: 50
  Reason: Property failed on iteration 23
```

Re-run with that seed to reproduce the failure.

## Test Framework

The tests use a custom property testing framework in `property_test.hpp`:

```cpp
auto result = CheckProperty("Test Name", [&](auto& gen) {
    // Generate random inputs
    auto value = gen.UniformFloat<double>(-10.0, 10.0);
    // Return true if property holds
    return std::isfinite(value);
}, 100, seed);
```

## Properties Tested

### Mathematical Invariants (PROP-01)
- **Matmul Identity**: A @ I = A
- **Transpose Involution**: (A^T)^T = A

### Algorithmic Correctness (PROP-02)
- **Sort**: Output is sorted
- **Reduce**: Associative operation order-independent
- **Scan**: Prefix sum correctness

### Numerical Stability (PROP-03)
- **Precision Consistency**: FP16/FP32/FP64 relationships
- **NaN Propagation**: NaN only when expected
- **Inf Propagation**: Inf only when expected

## Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| PROP-01 | Mathematical invariants | ✅ |
| PROP-02 | Algorithmic correctness | ✅ |
| PROP-03 | Numerical stability | ✅ |
| PROP-04 | Reproducible seeds | ✅ |
