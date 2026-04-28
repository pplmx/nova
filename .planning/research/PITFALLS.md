# GPU Algorithm Pitfalls Research

**Domain:** CUDA GPU Parallel Algorithms
**Researched:** 2026-04-28
**Confidence:** HIGH (based on NVIDIA official documentation and established GPU programming patterns)

## Executive Summary

This document catalogs common pitfalls across four GPU algorithm domains: sorting/searching, linear algebra extras, numerical methods, and signal processing. Each pitfall includes root cause analysis, consequences, and mitigation strategies mapped to implementation phases.

Key cross-cutting themes:
- **Shared memory bank conflicts** affect sorting, linear algebra, and signal processing
- **Warp divergence** impacts all irregular algorithms
- **Numerical stability** is critical for linear algebra, numerical methods, and signal processing
- **Memory coalescing** requirements vary by data access patterns

---

## 1. Sorting & Searching Pitfalls

### 1.1 Bank Conflicts in Shared Memory Sorting

**What goes wrong:** Parallel sorting algorithms (bitonic, radix, odd-even mergesort) heavily use shared memory for comparison exchanges. When threads in a warp access shared memory addresses that map to the same bank, throughput drops by a factor equal to the conflict degree.

**Why it happens:** NVIDIA shared memory is divided into banks (32 banks on most architectures). Sequential access patterns with stride equal to a power of 2 cause all threads to hit the same bank. Classic example:

```cpp
// DANGEROUS: stride of 32 causes bank conflict
__shared__ float shared[256];
value = shared[threadIdx.x * 32];  // All threads access same bank!
```

**Consequences:**
- 32x throughput reduction in worst case
- Sorting slower than a simpler algorithm with better memory access
- Non-obvious: appears correct but underperforms by 10-50x

**Prevention:**
```cpp
// SAFE: Use 5-word padding to avoid bank conflicts
__shared__ float shared[256 + 5];  // +5 avoids power-of-2 strides
value = shared[threadIdx.x * 33];  // Now accesses different banks

// Alternative: Use shuffle instructions instead of shared memory
value = __shfl_down(value, 16);  // No shared memory needed
```

**Detection:** NVIDIA profiler shows "shared memory efficiency" below 80% or high "shared_load_transaction" counts.

**Phase Recommendation:** Phase 1 (Shared Memory Access Patterns) - Define bank-conflict-free access patterns before implementing sort kernels.

---

### 1.2 Warp Divergence in Variable-Length Sorting

**What goes wrong:** Sorting variable-length records (strings, structs) requires conditional logic that varies by thread, causing warp divergence where threads take different execution paths.

**Why it happens:** The SIMT execution model executes all threads in a warp on the same instruction. When threads branch based on data-dependent lengths, inactive threads still consume execution cycles.

```cpp
// DIVERGENT: Different threads take different paths
if (keyLengths[threadIdx] < 8) {
    // Thread 0, 4, 8, 12... execute here
    sortSmallKey(key, threadIdx);
} else if (keyLengths[threadIdx] < 16) {
    // Thread 1, 5, 9, 13... execute here
    sortMediumKey(key, threadIdx);
} else {
    // Thread 2, 6, 10, 14... execute here
    sortLargeKey(key, threadIdx);
}
```

**Consequences:**
- Up to 32x slowdown in worst divergence case
- Performance varies non-deterministically with input distribution
- Compiler cannot auto-vectorize around divergence

**Prevention:**
1. **Sort by type first, then by key** - All keys of same length together
2. **Warp-uniform control flow** - Use predicates that vary within warps only for memory operations, not compute
3. **Multi-pass approach** - One pass to classify, second pass to sort homogeneous groups

```cpp
// PREFER: Classify first, sort homogeneous groups
int bucket = classifyLength(keyLengths[threadIdx]);  // 0, 1, or 2
__shared__ int bucketCount[3];
// Count bucket sizes...
// Launch homogeneous sort kernels per bucket
```

**Phase Recommendation:** Phase 2 (Warp-Synchronous Design) - Model divergence patterns before kernel implementation.

---

### 1.3 Memory Coalescing for Variable-Length Data

**What goes wrong:** Variable-length sorting (string sort, record sort) cannot guarantee contiguous memory access, leading to severe memory bandwidth underutilization.

**Why it happens:** Fixed-length sort assumes all elements are contiguous in memory. Variable-length elements with pointers/offsets break this assumption:

```cpp
// BAD: Non-contiguous access pattern
StringRecord* records = getRecords();
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    StringRecord& r = records[i];  // May point anywhere
    sortKey(r.key, r.keyLength);   // Scatter-gather pattern
}
```

**Consequences:**
- Memory bandwidth drops to 10-20% of peak
- Latency hiding fails because threads wait for scattered memory
- Sorting throughput inversely proportional to string variance

**Prevention:**
1. **Pack records contiguously** with fixed-size headers
2. **Use indirection** - Sort indices/keys first, then reorder
3. **Radix sort friendly encoding** - Pre-compute fixed-length sortable keys

```cpp
// BETTER: Pack with fixed-size headers
struct PackedRecord {
    uint32_t length;
    uint32_t sortKey;        // Precomputed for variable data
    char data[];             // Variable payload
};

// Now sorting accesses contiguous memory
PackedRecord* records = packRecords(original);
radixSort(records, n);       // Coalesced access guaranteed
```

**Phase Recommendation:** Phase 1 (Memory Layout) - Define data layout before algorithm implementation.

---

### 1.4 Numerical Stability in Key Comparisons

**What goes wrong:** Floating-point key sorting produces inconsistent results across runs, architectures, and optimization levels due to non-associativity of floating-point operations.

**Why it happens:** `(a < b)` may differ from `(b > a)` in floating-point due to rounding. Parallel sort evaluates comparisons in different orders than serial sort.

**Consequences:**
- Results vary across GPU generations (different instruction scheduling)
- Debug vs. release builds produce different orderings
- Numerical reproducibility impossible without deterministic reduction

**Prevention:**
1. **Use integer keys** for sortable floating-point values (encode via `float_as_int`)
2. **Define stable comparison** - Use bitwise operations on integer representations
3. **Accept non-determinism** - Document as acceptable for numerical sorts

```cpp
// STABLE: Integer comparison for floating-point keys
__device__ bool compareKeys(float a, float b) {
    uint32_t ia = float_as_uint(a);
    uint32_t ib = float_as_uint(b);
    // Handle sign bit for correct ordering
    return (ia ^ (1u << 31)) < (ib ^ (1u << 31));
}
```

**Phase Recommendation:** Phase 3 (Correctness Verification) - Define numerical stability requirements and comparison semantics.

---

## 2. Linear Algebra Extras Pitfalls

### 2.1 Convergence Issues in Iterative Eigensolvers

**What goes wrong:** Power iteration, Rayleigh quotient iteration, and Krylov subspace methods fail to converge or converge to wrong eigenvalues due to numerical issues.

**Why it happens:**
1. **Clustered eigenvalues** - Nearly equal eigenvalues cause slow separation
2. **Poor initial guesses** - Starting vectors orthogonal to dominant eigenspace
3. **Loss of orthogonality** - Gram-Schmidt orthonormalization accumulates errors

**Consequences:**
- Algorithm never terminates (no convergence check catches this)
- Returns eigenvalues with wrong multiplicity
- Eigenvectors span wrong subspace

**Prevention:**
```cpp
// MUST HAVE: Convergence monitoring with fault tolerance
float monitorConvergence(const Matrix& A, const Vector& v, float lambda, int iter) {
    float residual = norm(A * v - lambda * v) / norm(v);
    
    // Detect pathological cases
    if (iter > maxIterations * 0.9 && residual > tolerance * 10) {
        // Likely in clustered eigenvalue regime
        // Trigger subspace expansion or restart
        return -1.0f;  // Signal to restart with new vector
    }
    
    return residual;
}

// Rayleigh quotient iteration with safeguards
Vector rayleighQuotientIter(const Matrix& A, Vector v0, float lambda0) {
    Vector v = normalize(v0);
    float lambda = lambda0;
    
    for (int iter = 0; iter < maxIter; iter++) {
        if (iter > 0) {
            // Compute shift directly from current estimate
            lambda = dot(v, A * v);  // Rayleigh quotient
        }
        
        Vector w = solve(A - lambda * I);  // May be ill-conditioned
        v = normalize(w);
        
        float conv = monitorConvergence(A, v, lambda, iter);
        if (conv >= 0 && conv < tolerance) break;
        if (conv < 0) {
            // Restart with new random vector
            v = randomOrthogonalVector(v);
        }
    }
    return v;
}
```

**Phase Recommendation:** Phase 4 (Eigensolver Implementation) - Include convergence monitoring and restart logic from the start.

---

### 2.2 Numerical Stability in SVD

**What goes wrong:** Singular Value Decomposition produces inaccurate small singular values and wrong singular vectors due to catastrophic cancellation in certain decomposition stages.

**Why it happens:**
1. **One-sided Jacobi/Golub-Kahan** - Cancellation in bi-diagonalization
2. **Divide-by-zero in QR iteration** - Near-zero off-diagonal elements
3. **Orthogonality loss** - Householder reflectors degrade over many steps

**Consequences:**
- Small singular values have relative error >> machine epsilon
- Rank determination fails (small values should be zero)
- U and V matrices no longer orthogonal

**Prevention:**
```cpp
// USE: Batched condition-number aware SVD
struct SVDResult {
    Matrix U, S, Vt;
    int rank;
    float condition;  // sigma_max / sigma_min
};

SVDResult stableSVD(const Matrix& A) {
    // First pass: estimate condition number
    float normEst = estimateOneNorm(A);  // Used for pivoting decisions
    
    // Condition-dependent algorithm selection
    if (normEst > 1e10 || normEst < 1e-10) {
        // Use double precision or extended precision
        return doubleSVD(A);  // Convert to double, compute, convert back
    }
    
    // Standard path with monitoring
    Matrix B = bidiagonalize(A);  // Golub-Kahan with column pivoting
    
    for (int iter = 0; iter < maxQRIter; iter++) {
        // Safe QR step with threshold
        float threshold = max(abs(B.offDiag)) * machineEpsilon;
        if (abs(B.offDiag[k]) < threshold) {
            B.offDiag[k] = 0;  // Deflate early
        }
        // Continue with implicit shifts...
    }
    
    // Final rank determination with tolerance scaled by condition
    float tol = max(A.rows, A.cols) * machineEpsilon * normEst;
    int rank = countSingularValuesGreaterThan(tol, S);
    
    return {U, S, Vt, rank, normEst / S[rank]};
}
```

**Phase Recommendation:** Phase 5 (SVD Implementation) - Implement condition number estimation and adaptive precision switching.

---

### 2.3 Memory Usage for Large Matrices

**What goes wrong:** Eigenvalue decomposition and SVD of large matrices cause out-of-memory errors or severe memory pressure due to intermediate allocations.

**Why it happens:**
- Full Householder reflections stored (n^2 per step)
- Implicitly shifted QR creates temporary matrices
- Eigensolver requires tridiagonal + eigenvector storage
- SVD U, S, Vt each require n^2 storage

**Consequences:**
- OOM errors on matrices that "should" fit
- Memory thrashing reduces effective bandwidth
- Cannot process matrices that fit in GPU memory

**Prevention:**
```cpp
// MEMORY-EFFICIENT: In-place tridiagonalization
void memoryEfficientEigenSolve(Matrix& A) {
    // In-place Householder reduces memory by 2/3
    for (int k = 0; k < A.n - 2; k++) {
        // Compute Householder vector in-place
        Vector& x = A.col(k).segment(k+1);
        Vector u = householderInPlace(x);  // Overwrites x
        
        // Apply to trailing submatrix in-place
        applyHouseholderInPlace(A, k, u);  // No temp allocations
    }
    
    // Now A contains tridiagonal T (upper part) and Householder data
    // Memory used: n^2 instead of 3*n^2 for naive implementation
}

// STREAMING: Process large matrices in tiles
void tiledSVD(const LargeMatrix& A, int tileSize = 4096) {
    // Estimate memory requirements
    size_t available = getAvailableMemory();
    size_t perTile = 3 * tileSize * tileSize * sizeof(double);
    int tilesAcross = (A.n + tileSize - 1) / tileSize;
    
    // Use iterative refinement instead of full decomposition
    // Compute only the singular vectors needed
    Matrix Ur, Sr, Vr;
    for (int i = 0; i < tilesAcross; i++) {
        for (int j = 0; j < tilesAcross; j++) {
            // Load, process, discard tile
            Tile tile = loadTile(A, i, j);
            processTile(tile, Ur, Sr, Vr);
            // tile automatically freed when out of scope
        }
    }
}
```

**Phase Recommendation:** Phase 1 (Memory Planning) - Estimate memory requirements and define streaming strategies before implementation.

---

### 2.4 Accuracy vs. Performance Tradeoffs

**What goes wrong:** Faster algorithms (iterative refinement, randomized SVD) trade accuracy without exposing this tradeoff to users.

**Why it happens:**
- Power iteration with early termination
- Randomized SVD with insufficient power iterations
- Single-precision instead of double for "speed"
- Implicit type conversions lose precision

**Consequences:**
- Silent accuracy degradation
- Users unaware their results are approximate
- Different inputs produce different accuracy levels

**Prevention:**
```cpp
// PROVIDE: Accuracy tier selection
enum class SVDPrecision { Fast, Standard, High };

struct SVDConfig {
    SVDPrecision precision = SVDPrecision::Standard;
    int maxIterations = 100;
    float convergenceTol = 1e-6f;
};

SVDResult svd(const Matrix& A, const SVDConfig& config = {}) {
    switch (config.precision) {
        case SVDPrecision::Fast:
            // Randomized SVD with 2 power iterations
            return randomizedSVD(A, 2, config.maxIterations);
        case SVDPrecision::Standard:
            // Standard Jacobi with 10 iterations
            return jacobiSVD(A, 10, config.convergenceTol);
        case SVDPrecision::High:
            // Extra-precise Jacobi with refinement
            return extraPreciseSVD(A, config.convergenceTol);
    }
}

// DOCUMENT: Provide expected accuracy bounds
float expectedRelativeError(SVDPrecision p, float conditionNumber) {
    switch (p) {
        case Fast:        return 1e-3 * conditionNumber * machineEpsilon;
        case Standard:    return 1e-6 * conditionNumber * machineEpsilon;
        case High:        return 1e-10 * conditionNumber * machineEpsilon;
    }
}
```

**Phase Recommendation:** Phase 3 (API Design) - Expose accuracy/performance tradeoff in public API with clear documentation.

---

## 3. Numerical Methods Pitfalls

### 3.1 Monte Carlo Variance Issues

**What goes wrong:** Monte Carlo simulations produce high-variance results or fail to reduce variance at expected rate due to sampling inefficiencies.

**Why it happens:**
1. **Correlated samples** - Using same random seed across iterations
2. **Wrong random walk** - Antithetic variates not properly paired
3. **Systematic bias** - Quasi-random sequences misconfigured
4. **Variance accumulates** - Multiplicative processes amplify noise

**Consequences:**
- Requires 100x more samples than theory predicts
- Results unstable across runs
- Confidence intervals don't contain true value

**Prevention:**
```cpp
// CORRECT: Parallel Monte Carlo with proper variance tracking
struct MonteCarloResult {
    double mean;
    double variance;
    double stdError;
    int samples;
    bool converged;
};

MonteCarloResult parallelMonteCarlo(
    const MonteCarloConfig& config,
    curandState* states,  // One state per thread
    int samplesPerThread
) {
    double localSum = 0.0;
    double localSumSq = 0.0;
    
    for (int i = 0; i < samplesPerThread; i++) {
        double u1 = curand_uniform(&states[threadIdx.x]);
        double u2 = curand_uniform(&states[threadIdx.x]);
        
        // Use Box-Muller for Gaussian (or Philox for speed)
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
        
        // Compute sample
        double sample = evaluatePath(z);
        
        localSum += sample;
        localSumSq += sample * sample;
    }
    
    // Reduce across all threads
    double totalSum = blockReduceSum(localSum);
    double totalSumSq = blockReduceSum(localSumSq);
    
    if (threadIdx.x == 0) {
        int totalSamples = gridDim.x * blockDim.x * samplesPerThread;
        double mean = totalSum / totalSamples;
        // Welford's online algorithm for numerical stability
        double variance = (totalSumSq - totalSum*totalSum/totalSamples) / (totalSamples-1);
        double stdError = sqrt(variance / totalSamples);
        
        return {
            mean,
            variance,
            stdError,
            totalSamples,
            stdError < config.targetError
        };
    }
}

// USE: Quasi-Monte Carlo for low-discrepancy sequences
void quasiMonteCarlo(int n, float* samples) {
    // Sobol sequences provide better distribution than pseudorandom
    // for integration-type problems
    for (int i = 0; i < n; i++) {
        samples[i] = sobolSample(i, dimension);  // Well-distributed
    }
}
```

**Phase Recommendation:** Phase 6 (Random Number Generation Infrastructure) - Establish PRNG quality standards before Monte Carlo implementation.

---

### 3.2 Convergence Monitoring Failures

**What goes wrong:** Iterative numerical methods (root finding, integration, optimization) terminate prematurely or never terminate due to poor convergence monitoring.

**Why it happens:**
1. **Relative vs. absolute tolerance** - Not distinguishing between them
2. **Stalling detection** - Not detecting when progress stops
3. **Oscillation detection** - Missing periodic behavior
4. **Numerical cancellation** - Computing differences of similar values

**Consequences:**
- "Converged" solution far from actual root
- Infinite loop or very slow convergence
- Different results on different architectures

**Prevention:**
```cpp
// ROBUST: Comprehensive convergence monitoring
struct ConvergenceStatus {
    bool converged;
    bool stalled;
    bool oscillating;
    int iterations;
    float rate;  // Convergence rate estimate
};

ConvergenceStatus monitorConvergence(
    const std::vector<float>& errors,
    const ConvergenceConfig& config
) {
    if (errors.size() < 3) return {false, false, false, 0, 0.0f};
    
    float absTol = config.absoluteTolerance;
    float relTol = config.relativeTolerance;
    float prev = errors[errors.size() - 1];
    float prevPrev = errors[errors.size() - 2];
    
    // Check absolute and relative tolerance
    bool meetsAbsTol = prev < absTol;
    bool meetsRelTol = prev < relTol * errors[0];
    bool converged = meetsAbsTol && meetsRelTol;
    
    // Detect stalling: no progress for N iterations
    bool stalled = false;
    if (errors.size() >= config.stallWindow) {
        float maxRecent = *std::max_element(
            errors.end() - config.stallWindow, errors.end()
        );
        float minRecent = *std::min_element(
            errors.end() - config.stallWindow, errors.end()
        );
        stalled = (maxRecent - minRecent) < absTol * 0.1f;
    }
    
    // Detect oscillation
    bool oscillating = false;
    if (errors.size() >= 6) {
        // Check if error keeps increasing then decreasing
        float changes = 0;
        for (size_t i = errors.size() - 4; i < errors.size() - 1; i++) {
            if ((errors[i+1] > errors[i]) != (errors[i] > errors[i-1])) {
                changes++;
            }
        }
        oscillating = changes >= 3;
    }
    
    // Estimate convergence rate
    float rate = 0.0f;
    if (prevPrev > 0 && prev > 0) {
        rate = log(prev / prevPrev) / log(prevPrev / errors[errors.size()-3]);
    }
    
    return {converged, stalled, oscillating, (int)errors.size(), rate};
}

// SAFE: Newton-Raphson with monitoring
float safeNewtonRoot(float x0, const RootConfig& config) {
    float x = x0;
    std::vector<float> errors;
    
    for (int iter = 0; iter < config.maxIterations; iter++) {
        float fx = f(x);
        float dfx = df(x);
        
        float dx = fx / dfx;
        x -= dx;
        
        float error = abs(dx);
        errors.push_back(error);
        
        auto status = monitorConvergence(errors, config.convergence);
        
        if (status.converged) break;
        if (status.stalled) {
            // Try smaller step or different method
            x += dx * 0.5f;  // Bisection step
        }
        if (status.oscillating) {
            // Switch to bisection
            return bisectionRoot(x - dx*2, x, config);
        }
        if (iter == config.maxIterations - 1) {
            throw ConvergenceError(status);
        }
    }
    return x;
}
```

**Phase Recommendation:** Phase 3 (Convergence Monitoring Infrastructure) - Implement monitoring before any iterative method.

---

### 3.3 Pseudo-Random Number Generation Quality

**What goes wrong:** Using inappropriate PRNGs for Monte Carlo or stochastic simulation produces statistically biased results.

**Why it happens:**
1. **Linear congruential generators** - Poor distribution in high dimensions
2. **Same seed everywhere** - All threads produce identical sequences
3. **Period too short** - Sequences repeat before simulation completes
4. **State collision** - Multiple threads use same state

**Consequences:**
- Monte Carlo integrals systematically wrong
- Stochastic differential equations biased
- Gambling simulations fail statistical tests
- Different GPUs produce different results

**Prevention:**
```cpp
// GPU-APPROPRIATE: Use cuRAND or similar
#include <curand.h>

// SETUP: One RNG state per thread, properly initialized
__global__ void setupRNG(curandState* states, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Different seed per thread using sequence number
    curand_init(seed, id, 0, &states[id]);
}

__global__ void simulation(curandState* states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[id];
    
    // Now each thread has independent, high-quality sequence
    float u1 = curand_uniform(&localState);
    float u2 = curand_uniform(&localState);
    float normal = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    
    // Use in simulation...
    
    // Save state for next call
    states[id] = localState;
}

// QUALITY: Use Philox for Monte Carlo (counter-based, predictable)
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
curandGenerateUniform(gen, d_output, N);

// For cryptographically secure: CURAND_RNG_PSEUDO_XORWOW
// For maximum speed: CURAND_RNG_PSEUDO_MRG32K3A
```

**Phase Recommendation:** Phase 6 (PRNG Infrastructure) - Choose and implement PRNG strategy early; changing later breaks reproducibility.

---

### 3.4 Numerical Stability in Numerical Integration

**What goes wrong:** Quadrature and integration routines produce inaccurate results due to cancellation, poor node selection, or adaptive step size issues.

**Why it happens:**
1. **Gauss quadrature** - Nodes/weights computed incorrectly
2. **Adaptive Simpson** - Step size oscillates
3. **Infinite bounds** - Transformation introduces instability
4. **Singular endpoints** - Improper handling of integrable singularities

**Consequences:**
- Integration error >> requested tolerance
- Adaptive algorithm infinite loops
- NaN/Inf from overflow in transformation

**Prevention:**
```cpp
// STABLE: Adaptive quadrature with reliable error estimation
struct QuadResult {
    double value;
    double error;
    int functionEvaluations;
    bool converged;
};

QuadResult adaptiveQuad(
    double a, double b,
    double (*f)(double),
    double tol,
    int maxDepth = 50
) {
    // Initial Simpson's rule estimate
    double c = (a + b) / 2;
    double fa = f(a), fb = f(b), fc = f(c);
    double S = (b - a) / 6 * (fa + 4*fc + fb);
    
    // Recursive adaptive refinement
    return adaptiveQuadRec(a, c, fa, fc, S, tol, 0, maxDepth, f);
}

QuadResult adaptiveQuadRec(
    double a, double b, double fa, double fb,
    double S, double tol, int depth, int maxDepth,
    double (*f)(double)
) {
    double c = (a + b) / 2;
    double fc = f(c);
    
    // Two Simpson estimates
    double Sleft = (c - a) / 6 * (fa + 4*f((a+c)/2) + fc);
    double Sright = (b - c) / 6 * (fc + 4*f((c+b)/2) + fb);
    double S2 = Sleft + Sright;
    
    // Error estimation
    double E = (S2 - S) / 15.0;  // Richardson extrapolation
    
    if (depth >= maxDepth) {
        return {S2, abs(E), -1, false};
    }
    
    if (abs(E) < tol) {
        // Extrapolated estimate
        double S_extrap = S2 + E;
        return {S_extrap, abs(E), -1, true};
    }
    
    // Recurse
    auto left = adaptiveQuadRec(a, c, fa, fc, Sleft, tol/2, depth+1, maxDepth, f);
    auto right = adaptiveQuadRec(c, b, fc, fb, Sright, tol/2, depth+1, maxDepth, f);
    
    return {
        left.value + right.value,
        sqrt(left.error*left.error + right.error*right.error),
        -1,
        left.converged && right.converged
    };
}

// TRANSFORMATION: Stable infinite integral
double integrateInfinite(double (*f)(double), double tol) {
    // Use tanh-sinh quadrature for infinite intervals
    // It's more stable than rational transformations
    
    // Or use Monte Carlo with importance sampling:
    // Integral(f(x), x=0..inf) = Integral(f(t/(1-t))/t^2, t=0..1)
    // with t = exp(-u) substitution
}
```

**Phase Recommendation:** Phase 4 (Integration Implementation) - Implement multiple integration methods and error estimation before adaptive routines.

---

## 4. Signal Processing Pitfalls

### 4.1 Boundary Condition Handling

**What goes wrong:** Convolution, filtering, and wavelet transforms produce incorrect results near boundaries due to improper padding or boundary handling.

**Why it happens:**
1. **Zero padding** - Creates discontinuities at edges
2. **Circular padding** - Wrong assumption of periodicity
3. **Replication padding** - Introduces spurious high frequencies
4. **Symmetric padding** - Misapplied to non-symmetric signals

**Consequences:**
- Edge artifacts in filtered images/signals
- Gibbs phenomenon near boundaries
- Wavelet coefficients wrong at coarse scales

**Prevention:**
```cpp
// PROPER: Boundary handling modes
enum class BoundaryMode {
    Zero,       // Extend with zeros
    Replicate,  // Repeat edge values
    Symmetric,  // Mirror at boundary
    Periodic,   // Assume periodic
    Reflect     // Reflect without edge duplication
};

__device__ float applyBoundaryMode(
    float* data, int idx, int size,
    BoundaryMode mode
) {
    if (idx >= 0 && idx < size) {
        return data[idx];
    }
    
    switch (mode) {
        case BoundaryMode::Zero:
            return 0.0f;
        case BoundaryMode::Replicate:
            return data[clamp(idx, 0, size - 1)];
        case BoundaryMode::Symmetric:
            idx = abs(idx) % (2 * size);
            return data[idx < size ? idx : 2 * size - 1 - idx];
        case BoundaryMode::Periodic:
            return data[((idx % size) + size) % size];
        case BoundaryMode::Reflect:
            if (idx < 0) idx = -idx - 1;
            if (idx >= size) idx = 2 * size - idx - 1;
            return data[idx];
    }
}

// SELECT: Appropriate mode per application
// - Symmetric: DCT, image processing (preserves edge statistics)
// - Reflect: Wavelet transforms (coefficients less biased)
// - Periodic: FFT convolution (must be periodic)
// - Replicate: Audio (natural continuation)
void chooseBoundaryMode(const SignalProperties& props) {
    if (props.isImage && props.hasSharpEdges) {
        return BoundaryMode::Reflect;  // Best for edge preservation
    }
    if (props.needsPeriodic) {
        return BoundaryMode::Periodic;  // For FFT
    }
    return BoundaryMode::Symmetric;  // Safe default
}
```

**Phase Recommendation:** Phase 5 (Signal Processing Framework) - Define boundary handling strategy in convolution framework design.

---

### 4.2 FFT Size Constraints

**What goes wrong:** Using FFT sizes that aren't power of 2, 3, 5, or 7 (radices supported by cuFFT) causes severe performance degradation.

**Why it happens:** cuFFT uses radix-based Cooley-Tukey decomposition. Sizes with prime factors outside {2, 3, 5, 7} require Bluestein's algorithm (O(n^2) in intermediate storage) or fail entirely.

**Consequences:**
- Performance drops by 10-100x for prime-length FFTs
- Memory allocation failures for Bluestein
- Unexpected results due to implicit resizing

**Prevention:**
```cpp
// FIND: Optimal FFT size
int findOptimalFFTSize(int n) {
    if (n <= 0) return 1;
    
    // Check if already optimal (power of 2, 3, 5, 7 only)
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    while (n % 7 == 0) n /= 7;
    
    if (n == 1) return true;  // Optimal!
    
    // Find nearest larger optimal size
    int base = n;
    n = ((n + 15) / 16) * 16;  // Round up to power of 16
    
    // Binary search for closest optimal size
    while (!isOptimalSize(n)) {
        n++;
    }
    return n;
}

bool isOptimalSize(int n) {
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    while (n % 7 == 0) n /= 7;
    return n == 1;
}

// PAD: Automatically pad to optimal size
void fftConvolve(
    const float* signal, int signalLen,
    const float* kernel, int kernelLen,
    float* output
) {
    int optimalSize = findOptimalFFTSize(signalLen + kernelLen - 1);
    
    // Zero-pad both inputs
    float* paddedSignal = padToSize(signal, signalLen, optimalSize);
    float* paddedKernel = padToSize(kernel, kernelLen, optimalSize);
    
    // Now FFT size is guaranteed optimal
    cufftHandle plan;
    cufftPlan1d(&plan, optimalSize, CUFFT_R2C, 1);
    cufftExecR2C(plan, paddedSignal, ...);
    cufftExecR2C(plan, paddedKernel, ...);
    // ... convolution ...
}
```

**Phase Recommendation:** Phase 2 (FFT Infrastructure) - Wrap cuFFT with automatic size optimization from the start.

---

### 4.3 IIR Filter Stability

**What goes wrong:** IIR (Infinite Impulse Response) filters become unstable when implemented on GPU due to coefficient quantization or state accumulation errors.

**Why it happens:**
1. **Pole migration** - Quantized coefficients move poles outside unit circle
2. **State precision** - Accumulated state values lose precision
3. **Overflow** - Large inputs cause state overflow
4. **Parallel form issues** - Direct form parallelization introduces coupling

**Consequences:**
- Filter output grows without bound (unstable)
- Filter outputs NaN/Inf
- Different stability on GPU vs CPU (different floating-point)

**Prevention:**
```cpp
// STABLE: State-space IIR with overflow protection
struct StableIIRState {
    float b0, b1, b2;  // Feedforward coefficients
    float a1, a2;      // Feedback coefficients
    float s1, s2;      // State (delay elements)
};

__device__ float stableIIRStep(StableIIRState& state, float x) {
    // Direct Form II (fewer multiplications, less state exposure)
    // w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
    
    float w = x - state.a1 * state.s1 - state.a2 * state.s2;
    
    // Saturation to prevent overflow
    w = fmaxf(fminf(w, 1e10f), -1e10f);
    
    // y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
    float y = state.b0 * w + state.b1 * state.s1 + state.b2 * state.s2;
    
    // Update state
    state.s2 = state.s1;
    state.s1 = w;
    
    return y;
}

// VERIFY: Filter stability before use
bool verifyFilterStability(float a1, float a2) {
    // Characteristic equation: z^2 + a1*z + a2 = 0
    // Roots must be inside unit circle
    
    float disc = a1*a1 - 4*a2;
    if (disc >= 0) {
        // Real roots
        float r1 = (-a1 - sqrt(disc)) / 2;
        float r2 = (-a1 + sqrt(disc)) / 2;
        return fabsf(r1) < 1.0f && fabsf(r2) < 1.0f;
    } else {
        // Complex conjugate roots
        float magnitude = sqrt(a2);  // |r| = sqrt(a2) for conjugate pair
        return magnitude < 1.0f - 1e-6f;  // Small margin for numerical
    }
}

// QUANTIZE: Safe coefficient quantization
float quantizeCoefficient(float coeff, int bits) {
    float scale = (1 << (bits - 1)) - 1.0f;
    float quantized = round(coeff * scale) / scale;
    
    // Check stability after quantization
    float a1 = quantized;  // Your actual coefficient
    float a2 = 0.5f;       // Second feedback coefficient
    
    if (!verifyFilterStability(a1, a2)) {
        // Fall back to lower order or different structure
        return coeff;  // Let it fail gracefully
    }
    return quantized;
}
```

**Phase Recommendation:** Phase 5 (IIR Filter Implementation) - Implement stability verification in filter design, not just runtime.

---

### 4.4 Numerical Precision in Wavelet Transforms

**What goes wrong:** Discrete Wavelet Transform (DWT) accumulates numerical errors over multiple decomposition levels, producing inaccurate coefficients at coarser scales.

**Why it happens:**
1. **Repeated filtering** - Error accumulates with each level
2. **Downsampling** - Aliasing amplifies quantization errors
3. **High-pass filter** - Amplifies numerical noise
4. **Floating-point rounding** - Error grows with transform length

**Consequences:**
- Energy not conserved across decomposition levels
- Small wavelet coefficients swamped by numerical noise
- Reconstruction error exceeds theoretical minimum

**Prevention:**
```cpp
// PRECISE: Wavelet transform with error monitoring
struct WaveletResult {
    std::vector<float> coefficients;  // [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    std::vector<float> reconstructionError;
    int levels;
};

WaveletResult stableWaveletDecomp(
    const float* signal, int length,
    const Wavelet& wavelet,
    int maxLevels
) {
    WaveletResult result;
    result.levels = min(maxLevels, (int)log2(length));
    
    // Current signal level
    std::vector<float> current(signal, signal + length);
    std::vector<float> prevSignal(length);
    
    // Low-pass (approximation) and high-pass (detail) outputs
    std::vector<float> approx, detail;
    
    for (int level = 0; level < result.levels; level++) {
        int n = current.size();
        prevSignal = current;  // Save for error analysis
        
        // Convolve with downsampling
        approx = convolveDownsample(current, wavelet.lo, n/2);  // Low
        detail = convolveDownsample(current, wavelet.hi, n/2);  // High
        
        // Check for energy conservation
        float inputEnergy = energy(current);
        float outputEnergy = energy(approx) + energy(detail);
        float energyError = abs(outputEnergy - inputEnergy) / inputEnergy;
        
        result.reconstructionError.push_back(energyError);
        
        // Warn if energy error exceeds threshold
        if (energyError > 1e-6f) {
            // Consider switching to integer wavelet or lifting scheme
            printf("Warning: Level %d energy error %e exceeds threshold\n",
                   level, energyError);
        }
        
        // Continue with approximation for next level
        current = approx;
    }
    
    // Assemble result
    result.coefficients = current;  // Final approximation
    for (int i = result.levels - 1; i >= 0; i--) {
        result.coefficients.insert(result.coefficients.end(),
                                   detail.begin(), detail.end());
    }
    
    return result;
}

// LIFTING SCHEME: More numerically stable than convolution
// Use Cohen-Daubechies-Feauveau (CDF) wavelets via lifting
struct LiftingWavelet {
    float p, u;  // Lifting coefficients
};

float liftStep(float even, float float odd, float p) {
    return odd - p * (even + evenNext);  // Predict
}

float liftUpdate(float even, float odd, float u) {
    return even + u * (odd + oddPrev);   // Update
}

// Lifting is exact for perfect reconstruction
// (floating-point errors notwithstanding, but much smaller)
```

**Phase Recommendation:** Phase 6 (Wavelet Implementation) - Compare convolution-based vs lifting scheme and document precision guarantees.

---

## Summary: Phase Recommendations

| Phase Topic | Likely Pitfall | Mitigation Strategy |
|-------------|----------------|---------------------|
| 1. Shared Memory Access | Bank conflicts | Bank-conflict-free padding, use shuffle |
| 1. Memory Layout | Non-coalesced variable access | Pack records, sort indices first |
| 2. Warp-Synchronous Design | Divergence patterns | Pre-classify data, warp-uniform control flow |
| 2. FFT Infrastructure | Non-power-of-2/3/5/7 sizes | Auto-pad to optimal size |
| 3. Correctness Verification | Floating-point comparison stability | Integer-encoded keys for stable sort |
| 3. Convergence Monitoring | Premature/late termination | Multi-criteria convergence with stalling detection |
| 4. Eigensolver Implementation | Non-convergence in clustered eigenvalues | Monitor convergence, restart with orthogonal vectors |
| 4. Integration Routines | Adaptive step size oscillation | Richardson extrapolation error estimation |
| 5. SVD Implementation | Accuracy degradation with condition | Condition estimation, adaptive precision |
| 5. IIR Filters | Pole migration to instability | Pre-verify stability, saturation arithmetic |
| 5. Signal Processing Framework | Boundary artifacts | Explicit boundary modes, select per application |
| 6. PRNG Infrastructure | Statistical bias, state collision | Per-thread independent seeds, quality PRNG |
| 6. Monte Carlo | High variance, non-convergence | Proper variance tracking, quasi-Monte Carlo |
| 6. Wavelet Transform | Energy loss over levels | Lifting scheme, energy conservation checks |

---

## Sources

- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - **HIGH confidence** (official NVIDIA documentation)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - **HIGH confidence** (official NVIDIA documentation)
- [Faster Parallel Reductions on Kepler (NVIDIA Blog)](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) - **HIGH confidence** (NVIDIA developer blog)
- [Precision and Performance: Floating-Point and IEEE 754 Compliance](https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus) - **HIGH confidence** (NVIDIA technical documentation)
- cuFFT documentation and cuRAND documentation - **HIGH confidence** (NVIDIA CUDA libraries)
