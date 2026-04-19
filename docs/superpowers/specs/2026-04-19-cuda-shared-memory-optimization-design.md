# CUDA Shared Memory Optimization - Design Spec

## Project Overview

- **Project**: 基于 shared memory 的矩阵乘法优化
- **目标**: 学习 CUDA 内存优化技巧,对现有矩阵乘法进行性能优化
- **产出**: 优化后的 matrix_mult.cu + benchmark 对比

## Architecture

### 组件

1. **Naive 版本** (已有) - 每个线程计算一个输出元素,直接访问 global memory
2. **Tiled 版本** (新增) - 使用 shared memory 做 tile,减少 global memory 访问
3. **Benchmarks** (新增) - 对比两种实现的性能

### 优化技术点

| 技术 | 描述 |
|------|------|
| Tiling | 将矩阵分块加载到 shared memory |
| Memory Coalescing | 确保 global memory 访问是合并的 |
| Bank Conflict 规避 | 通过 padding 避免 shared memory bank conflict |

## Data Flow

```
Host (CPU) → cudaMalloc (GPU) → Kernel (compute) → cudaMemcpy (Host)
                 ↑                                           ↓
                 └───────────────────────────────────────────┘
```

Tiled 版本流程:
1. 每个 block 将 A 和 B 的 tile 加载到 shared memory
2. Block 内线程合作计算 partial sum
3. 结果写入 global memory

## Implementation Details

### 文件结构

```
include/
  matrix_mult.h      # 添加 tiled 版本函数声明

src/
  matrix_mult.cu     # 添加 tiled kernel 和 benchmark 代码
  main.cpp           # 添加 benchmark 入口
```

### 新增函数

```cpp
// Tiled 矩阵乘法 (带 shared memory 优化)
template <typename T>
void multiplyMatricesTiled(const T* h_A, const T* h_B, T* h_C, int M, int N, int K);

// Benchmark 函数
void benchmarkMatrixMultiplication(int size);
```

### Kernel 设计

- **Block size**: 16x16 (保持一致)
- **Tile size**: 16x16 (与 block 大小相同)
- **使用 __shared__ 声明 shared memory 数组

## Testing

1. **正确性验证**: 优化版本与 naive 版本结果一致
2. **性能测试**: 对比不同矩阵大小下的执行时间
3. **矩阵大小**: 512x512, 1024x1024, 2048x2048

## Acceptance Criteria

- [ ] Tiled 版本输出结果与 naive 版本一致
- [ ] Tiled 版本在 1024x1024 及以上规模有显著加速 (>1.5x)
- [ ] 代码可编译通过,无 warnings
