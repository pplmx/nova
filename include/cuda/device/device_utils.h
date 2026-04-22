#pragma once

#include <cstddef>

#include "error.h"

namespace cuda::device {

    enum class ReduceOp { SUM, MAX, MIN };

    constexpr int WARP_SIZE = 32;

    template <typename T>
    __device__ T warp_reduce(T val, ReduceOp op) {
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (op == ReduceOp::SUM) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            } else if (op == ReduceOp::MAX) {
                val = max(val, __shfl_down_sync(0xffffffff, val, offset));
            } else {
                val = min(val, __shfl_down_sync(0xffffffff, val, offset));
            }
        }
        return val;
    }

    template <typename T>
    __device__ T block_reduce(T val, ReduceOp op, T* shared_mem) {
        const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        const int block_size = blockDim.x * blockDim.y * blockDim.z;

        shared_mem[tid] = val;
        __syncthreads();

        for (int s = block_size / 2; s > WARP_SIZE; s /= 2) {
            if (tid < s) {
                if (op == ReduceOp::SUM) {
                    shared_mem[tid] += shared_mem[tid + s];
                } else if (op == ReduceOp::MAX) {
                    shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
                } else {
                    shared_mem[tid] = min(shared_mem[tid], shared_mem[tid + s]);
                }
            }
            __syncthreads();
        }

        if (tid < WARP_SIZE) {
            val = shared_mem[tid];
            val = warp_reduce(val, op);
        }

        return val;
    }

    template <typename T>
    __device__ T load_tile(const T* global, int idx, int max_idx, T default_val = T{}) {
        return idx < max_idx ? global[idx] : default_val;
    }

    template <typename T>
    __device__ void store_tile(T* global, int idx, T val, int max_idx) {
        if (idx < max_idx) {
            global[idx] = val;
        }
    }

}  // namespace cuda::device
