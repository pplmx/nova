/**
 * @file image_processing.cpp
 * @brief Image processing example demonstrating blur, sobel, and morphology
 * @example
 *
 * Compile:
 *   g++ -std=c++23 -I include examples/image_processing.cpp \
 *       -L build/lib -lcuda_impl -lcudart -o image_processing
 *
 * Run:
 *   ./image_processing --input image.pgm --output result.pgm --kernel sobel
 */

#include <image/types.hpp>
#include <image/gaussian_blur.hpp>
#include <image/sobel_edge.hpp>
#include <image/morphology.hpp>
#include <cuda/error/cuda_error.hpp>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

struct Args {
    const char* input = "input.pgm";
    const char* output = "output.pgm";
    const char* kernel = "sobel";
    int iterations = 1;
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --input <file>     Input image (PGM format)\n");
    printf("  --output <file>    Output image (PGM format)\n");
    printf("  --kernel <name>     Kernel: sobel, blur, morphology\n");
    printf("  --iterations <n>   Number of iterations\n");
}

int main(int argc, char** argv) {
    Args args;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args.input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args.output = argv[++i];
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            args.kernel = argv[++i];
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            args.iterations = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("Nova Image Processing Example\n");
    printf("Input: %s, Output: %s, Kernel: %s\n",
           args.input, args.output, args.kernel);

    // Load image
    nova::image::PGMImage input_img;
    if (!nova::image::load_pgm(args.input, input_img)) {
        fprintf(stderr, "Error: Cannot load image %s\n", args.input);
        return 1;
    }

    printf("Image size: %dx%d\n", input_img.width, input_img.height);

    // Create output image
    nova::image::PGMImage output_img(input_img.width, input_img.height);
    nova::image::PGMImage temp_img(input_img.width, input_img.height);

    // Create CUDA buffers
    nova::memory::Buffer<unsigned char> d_input(input_img.width * input_img.height);
    nova::memory::Buffer<unsigned char> d_output(input_img.width * input_img.height);

    // Copy input to device
    d_input.copy_from_host(input_img.data.data());

    // Apply kernel
    for (int i = 0; i < args.iterations; i++) {
        if (strcmp(args.kernel, "sobel") == 0) {
            // Sobel edge detection
            nova::image::sobel_edge(
                d_input.device_data(),
                d_output.device_data(),
                input_img.width,
                input_img.height,
                temp_img.width
            );
        } else if (strcmp(args.kernel, "blur") == 0) {
            // Gaussian blur
            nova::image::gaussian_blur(
                d_input.device_data(),
                d_output.device_data(),
                input_img.width,
                input_img.height,
                3,  // kernel size
                1.0 // sigma
            );
        } else if (strcmp(args.kernel, "morphology") == 0) {
            // Binary morphology (dilation)
            nova::image::morphology_dilate(
                d_input.device_data(),
                d_output.device_data(),
                input_img.width,
                input_img.height,
                nullptr // use default 3x3 kernel
            );
        }

        // Swap buffers for next iteration
        if (i < args.iterations - 1) {
            d_input.copy_from_device(d_output.device_data(),
                                    input_img.width * input_img.height);
        }
    }

    // Copy result to host
    d_output.copy_to_host(output_img.data.data());

    // Save output
    if (!nova::image::save_pgm(args.output, output_img)) {
        fprintf(stderr, "Error: Cannot save image %s\n", args.output);
        return 1;
    }

    printf("Processed image saved to %s\n", args.output);
    return 0;
}
