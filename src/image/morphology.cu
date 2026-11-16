#include <cuda_runtime.h>

#include "cuda/device/device_utils.h"
#include "image/morphology.h"

__global__ __launch_bounds__(256, 2) void sharpenKernel(const uint8_t* input, uint8_t* output, size_t width, size_t height, float strength) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height || x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
        return;
    }

    // Unsharp mask: out = in + strength * (in - blur), computed per channel because
    // the neighbours differ between channels.
    //
    // The kernel this replaced summed all eight neighbours with a centre coefficient
    // of 5 — that centre belongs to the 4-neighbour stencil, so the weights summed to
    // -3 instead of 0 for the Laplacian part, and the `/ 3` in the update was a
    // compensation that left the response at 2x. A uniform region of value c came out
    // as 2c (measured: 100 -> 200): flat areas were brightened and nothing was
    // sharpened.
    //
    // Here the Laplacian is the standard 4-neighbour operator, whose weights sum to
    // zero so a uniform region contributes nothing:
    //     [ 0  1  0 ]
    //     [ 1 -4  1 ]
    //     [ 0  1  0 ]
    for (int c = 0; c < 3; ++c) {
        const size_t idx = (y * width + x) * 3 + c;

        float neighbours = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx != 0 && dy != 0) {
                    continue;  // diagonals are not part of the 4-neighbour operator
                }
                if (dx == 0 && dy == 0) {
                    continue;  // the centre is handled below with its -4 weight
                }
                neighbours +=
                    static_cast<float>(input[((y + dy) * width + (x + dx)) * 3 + c]);
            }
        }

        const float center = static_cast<float>(input[idx]);
        const float laplacian = neighbours - 4.0f * center;

        // Subtracting the Laplacian sharpens; strength scales the effect, and 0 gives
        // the identity.
        float val = center - strength * laplacian;
        val = max(0.0f, min(255.0f, val));
        output[idx] = static_cast<uint8_t>(val);
    }
}

void sharpenImage(const uint8_t* d_input, uint8_t* d_output, size_t width, size_t height, float strength) {
    if (width == 0 || height == 0) {
        return;
    }
    // sharpenKernel only writes interior pixels: it returns early on the first and
    // last row and column, where the 3x3 stencil would read outside the image. Those
    // border pixels were left holding whatever the output allocation contained, and
    // the caller read them back. Copying the input first makes the border the
    // unmodified image, which is what "sharpening" means at an edge. Detected with
    // compute-sanitizer --tool initcheck, which reported 66 uninitialized accesses on
    // a 64x48 test image whose output was then copied to the host.
    CUDA_CHECK(cudaMemcpy(d_output, d_input, width * height * 3, cudaMemcpyDeviceToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sharpenKernel<<<grid, block>>>(d_input, d_output, width, height, strength);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ __launch_bounds__(256, 2) void thresholdKernel(const uint8_t* input, uint8_t* output, size_t width, size_t height, uint8_t threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = width * height * 3;

    if (idx < total) {
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

void applyThreshold(const uint8_t* d_input, uint8_t* d_output, size_t width, size_t height, uint8_t threshold) {
    if (width == 0 || height == 0) {
        return;
    }
    size_t total = width * height * 3;
    int block = 256;
    int grid = (total + block - 1) / block;

    thresholdKernel<<<grid, block>>>(d_input, d_output, width, height, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ __launch_bounds__(256, 2) void erodeKernel(const uint8_t* input, uint8_t* output, size_t width, size_t height, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint8_t minVal = 255;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            size_t nx = min(max((int)x + dx, 0), (int)width - 1);
            size_t ny = min(max((int)y + dy, 0), (int)height - 1);
            size_t idx = (ny * width + nx) * 3;

            for (int c = 0; c < 3; ++c) {
                minVal = min(minVal, input[idx + c]);
            }
        }
    }

    size_t outIdx = (y * width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        output[outIdx + c] = minVal;
    }
}

void erodeImage(const uint8_t* d_input, uint8_t* d_output, size_t width, size_t height, int kernel_size) {
    if (width == 0 || height == 0) {
        return;
    }
    int half = kernel_size / 2;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    erodeKernel<<<grid, block>>>(d_input, d_output, width, height, half);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ __launch_bounds__(256, 2) void dilateKernel(const uint8_t* input, uint8_t* output, size_t width, size_t height, int half) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint8_t maxVal = 0;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            size_t nx = min(max((int)x + dx, 0), (int)width - 1);
            size_t ny = min(max((int)y + dy, 0), (int)height - 1);
            size_t idx = (ny * width + nx) * 3;

            for (int c = 0; c < 3; ++c) {
                maxVal = max(maxVal, input[idx + c]);
            }
        }
    }

    size_t outIdx = (y * width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        output[outIdx + c] = maxVal;
    }
}

void dilateImage(const uint8_t* d_input, uint8_t* d_output, size_t width, size_t height, int kernel_size) {
    if (width == 0 || height == 0) {
        return;
    }
    int half = kernel_size / 2;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    dilateKernel<<<grid, block>>>(d_input, d_output, width, height, half);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void openingImage(const uint8_t* d_input, uint8_t* d_output, size_t width, size_t height, int kernel_size) {
    uint8_t* temp;
    CUDA_CHECK(cudaMalloc(&temp, width * height * 3 * sizeof(uint8_t)));

    erodeImage(d_input, temp, width, height, kernel_size);
    dilateImage(temp, d_output, width, height, kernel_size);

    CUDA_CHECK(cudaFree(temp));
}

void closingImage(const uint8_t* d_input, uint8_t* d_output, size_t width, size_t height, int kernel_size) {
    uint8_t* temp;
    CUDA_CHECK(cudaMalloc(&temp, width * height * 3 * sizeof(uint8_t)));

    dilateImage(d_input, temp, width, height, kernel_size);
    erodeImage(temp, d_output, width, height, kernel_size);

    CUDA_CHECK(cudaFree(temp));
}
