# Features Research

**Domain:** CUDA Library Enhancements

## Table Stakes (Must Have)

These are expected by any serious CUDA compute library:

### 1. Device-Aware Configuration

- Auto-detect compute capability
- Tune block size based on occupancy analysis
- Configurable grid-stride loops for large datasets
- Graceful degradation on older GPUs

### 2. Memory Management

- Memory pool with allocation tracking
- Pinned memory for async transfers
- Memory usage queries (`cudaMemGetInfo`)
- Proper memory alignment (256-byte aligned for L2)

### 3. Error Handling

- Rich error context (operation name, sizes, device info)
- Validation at API boundaries
- Informative exception messages

### 4. Testing & Benchmarking

- Unit tests for correctness
- Benchmark suite with consistent metrics
- Performance regression detection

## Differentiators (Competitive Advantage)

### FFT Implementation

**Why valuable:** Signal processing, image processing, convolution acceleration

**API approach (FFTW-inspired):**
```cpp
// Plan-based API similar to FFTW
class FFTPlan {
    int n;          // Transform size
    bool inverse;   // Forward/backward
    // Internal buffers
};

void fft_execute(FFTPlan* plan, const float* in, float* out);
void fft_destroy(FFTPlan* plan);
```

**Implementation options:**
1. cuFFT (NVIDIA library) — easiest, most performant
2. Custom FFT — more control, educational
3. Mixed — use cuFFT, wrap in our API

### Ray Tracing Primitives

**Why valuable:** Rendering, physics simulation, collision detection

**Core primitives:**
```cpp
struct Ray { float3 origin, dir; };
struct AABB { float3 min, max; };

// Intersections
bool ray_box_intersect(const Ray& r, const AABB& box, float& t);
bool ray_sphere_intersect(const Ray& r, const float3& center, float radius, float& t);

// BVH helpers
struct BVHNode { AABB bounds; int left, right, primitive; };
```

### Graph Algorithms

**Why valuable:** Social networks, recommendation systems, pathfinding

**Key algorithms:**
```cpp
// BFS with frontier management
void graph_bfs(const int* edges, int num_vertices, int start, int* distances);

// PageRank iteration
void pagerank_iteration(const float* adjacency, float* ranks, float damping, int n);
```

**Storage formats:**
- CSR (Compressed Sparse Row) — efficient for sparse graphs
- COO (Coordinate) — easy construction

### Neural Network Primitives

**Why valuable:** ML acceleration, transformer inference

**Core operations:**
```cpp
// Matrix multiply (with optional bias)
void matmul(const float* A, const float* B, float* C, int M, int N, int K, bool relu=false);

// Softmax (stable numerically)
void softmax(float* data, int N, int stride);

// Layer normalization
void layer_norm(const float* input, float* output, int N, int D, const float* gamma, const float* beta);

// ReLU variants
void relu(float* data, int N, float alpha=0.0f);  // Leaky ReLU option
```

## Anti-Features (Deliberately Not Building)

| Feature | Why Excluded |
|---------|--------------|
| Full ray tracer | Outside scope, requires scene management |
| Automatic differentiation | Complex, separate project |
| Python bindings | Different skill set, separate project |
| Distributed computing | Network complexity, single GPU focus |
| Real-time video pipeline | Requires streaming foundation first |
