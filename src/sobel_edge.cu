#include "sobel_edge.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void sobelKernel(const uint8_t* input, uint8_t* output,
                            size_t width, size_t height, float threshold) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int gx = -static_cast<int>(input[(y-1)*width+(x-1)]) + static_cast<int>(input[(y-1)*width+(x+1)])
             -2*static_cast<int>(input[y*width+(x-1)]) + 2*static_cast<int>(input[y*width+(x+1)])
             -static_cast<int>(input[(y+1)*width+(x-1)]) + static_cast<int>(input[(y+1)*width+(x+1)]);

    int gy = -static_cast<int>(input[(y-1)*width+(x-1)]) - 2*static_cast<int>(input[(y-1)*width+x]) - static_cast<int>(input[(y-1)*width+(x+1)])
             +static_cast<int>(input[(y+1)*width+(x-1)]) + 2*static_cast<int>(input[(y+1)*width+x]) + static_cast<int>(input[(y+1)*width+(x+1)]);

    int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx*gx + gy*gy)));

    output[y * width + x] = (magnitude > static_cast<int>(threshold)) ? 255 : 0;
}

void sobelEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                        size_t width, size_t height,
                        float threshold) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sobelKernel<<<grid, block>>>(d_input, d_output, width, height, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
