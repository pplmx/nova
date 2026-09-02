/**
 * @file image_processing.cpp
 * @brief Image processing example: Sobel edge, Gaussian blur, morphology
 *        dilation — ported to the current cuda::* / plain image APIs (TASK-064;
 *        the nova::image::* API and PGMImage/load_pgm/save_pgm this example
 *        used no longer exist).
 * @example
 *
 * Compile (part of the example targets):
 *   cmake --build build --target image_processing
 *
 * Run:
 *   ./build/bin/image_processing --input image.pgm --output result.pgm --kernel sobel
 */

#include <image/sobel_edge.h>
#include <image/gaussian_blur.h>
#include <image/morphology.h>
#include <cuda/memory/buffer.h>
#include <cuda/memory/buffer-inl.h>
#include <cuda/device/error.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct PGMImage {
    size_t width = 0;
    size_t height = 0;
    std::vector<unsigned char> data;
};

// Minimal PGM binary (P5) reader — the library has no image-IO helper, so the
// example owns one (grayscale, 8-bit, maxval 255).
bool load_pgm(const std::string& path, PGMImage& img) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "Error: cannot open %s\n", path.c_str());
        return false;
    }
    std::string magic;
    in >> magic;
    if (magic != "P5") {
        fprintf(stderr, "Error: %s is not a P5 (binary grayscale) PGM\n",
                path.c_str());
        return false;
    }
    int width = 0, height = 0, maxval = 0;
    in >> width >> height >> maxval;
    in.ignore();  // single whitespace after maxval
    if (width <= 0 || height <= 0 || maxval != 255) {
        fprintf(stderr,
                "Error: unsupported PGM header (expected 8-bit, maxval 255)\n");
        return false;
    }
    img.width = static_cast<size_t>(width);
    img.height = static_cast<size_t>(height);
    img.data.assign(static_cast<size_t>(width) * height, 0);
    in.read(reinterpret_cast<char*>(img.data.data()),
            static_cast<std::streamsize>(img.data.size()));
    if (in.gcount() != static_cast<std::streamsize>(img.data.size())) {
        fprintf(stderr, "Error: %s is truncated (expected %zu pixel bytes)\n",
                path.c_str(), img.data.size());
        return false;
    }
    return true;
}

bool save_pgm(const std::string& path, const PGMImage& img) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "Error: cannot write %s\n", path.c_str());
        return false;
    }
    out << "P5\n" << img.width << ' ' << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()),
              static_cast<std::streamsize>(img.data.size()));
    return out.good();
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -h, --help         Show this help and exit\n");
    printf("  --input <file>     Input image (PGM, P5/8-bit)\n");
    printf("  --output <file>    Output image (PGM)\n");
    printf("  --kernel <name>    Kernel: sobel, blur, morphology\n");
    printf("  --iterations <n>   Number of iterations (positive int)\n");
}

// Returns 1 on --help (exit 0), 0 on success, -1 on error (exit 1).
int parse_args(int argc, char** argv, std::string& input, std::string& output,
               std::string& kernel, int& iterations) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernel = argv[++i];
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v <= 0 || v > 100000) {
                fprintf(stderr, "Error: --iterations expects a positive "
                                "integer, got '%s'\n", argv[i]);
                return -1;
            }
            iterations = static_cast<int>(v);
        } else {
            fprintf(stderr, "Error: unknown or incomplete option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    // Validate the kernel up front: the old example silently no-op'ed on an
    // unknown --kernel and saved an uninitialized output image (ISS-003).
    if (kernel != "sobel" && kernel != "blur" && kernel != "morphology") {
        fprintf(stderr, "Error: unknown kernel '%s' (choose sobel | blur | "
                        "morphology)\n", kernel.c_str());
        return -1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input = "input.pgm";
    std::string output = "output.pgm";
    std::string kernel = "sobel";
    int iterations = 1;

    const int parse = parse_args(argc, argv, input, output, kernel, iterations);
    if (parse < 0) return 1;
    if (parse > 0) return 0;

    PGMImage in_img;
    if (!load_pgm(input, in_img)) return 1;
    printf("Loaded %s: %zux%zu\n", input.c_str(), in_img.width, in_img.height);

    const size_t npix = in_img.width * in_img.height;
    PGMImage out_img = in_img;  // keeps dims; data overwritten below
    out_img.data.resize(npix);

    try {
        cuda::memory::Buffer<unsigned char> d_in(npix);
        cuda::memory::Buffer<unsigned char> d_out(npix);
        d_in.copy_from(in_img.data.data(), npix);

        if (kernel == "sobel") {
            // sobelEdgeDetection is an RGB (3 bytes/pixel) kernel, while the
            // PGM input is grayscale — expand to RGB, run it, take the R
            // channel back to grayscale for the output.
            const size_t npix3 = npix * 3;
            std::vector<unsigned char> h_rgb(npix3);
            for (size_t i = 0; i < npix; ++i) {
                h_rgb[3 * i] = h_rgb[3 * i + 1] = h_rgb[3 * i + 2] =
                    in_img.data[i];
            }
            cuda::memory::Buffer<unsigned char> d_rgb_in(npix3);
            cuda::memory::Buffer<unsigned char> d_rgb_out(npix3);
            d_rgb_in.copy_from(h_rgb.data(), npix3);
            for (int i = 0; i < iterations; ++i) {
                sobelEdgeDetection(d_rgb_in.data(), d_rgb_out.data(),
                                   in_img.width, in_img.height);
                if (i < iterations - 1) {
                    CUDA_CHECK(cudaMemcpy(d_rgb_in.data(), d_rgb_out.data(),
                                          npix3, cudaMemcpyDeviceToDevice));
                }
            }
            std::vector<unsigned char> h_rgb_out(npix3);
            d_rgb_out.copy_to(h_rgb_out.data(), npix3);
            for (size_t i = 0; i < npix; ++i) {
                out_img.data[i] = h_rgb_out[3 * i];
            }
        } else {
            // Apply the (grayscale) kernel `iterations` times, swapping in/out.
            for (int i = 0; i < iterations; ++i) {
                if (kernel == "blur") {
                    cuda::algo::gaussianBlur(d_in, d_out, in_img.width,
                                             in_img.height,
                                             /*sigma=*/1.0f, /*kernel_size=*/3);
                } else {
                    dilateImage(d_in.data(), d_out.data(), in_img.width,
                                in_img.height);
                }
                if (i < iterations - 1) {
                    CUDA_CHECK(cudaMemcpy(d_in.data(), d_out.data(), npix,
                                          cudaMemcpyDeviceToDevice));
                }
            }
            d_out.copy_to(out_img.data.data(), npix);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    if (!save_pgm(output, out_img)) return 1;
    printf("Processed image saved to %s (%s, %d iteration%s)\n",
           output.c_str(), kernel.c_str(), iterations,
           iterations == 1 ? "" : "s");
    return 0;
}
