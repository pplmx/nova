#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @file morphology.h
 * @brief Image filters over packed RGB data.
 *
 * Every function here treats its buffers as PACKED RGB: the element count is
 * width * height * 3, not width * height. The kernels index (y * width + x) * 3, so a
 * caller that allocates width * height bytes gets a silent over-read — measured as a
 * 64-byte buffer accessed up to byte 192 with no CUDA error reported, because the
 * overrun landed inside the allocator's padding.
 *
 * A zero width or height is a no-op rather than a launch failure.
 */

void sharpenImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  float strength = 1.0f);

void applyThreshold(const uint8_t* d_input, uint8_t* d_output,
                    size_t width, size_t height,
                    uint8_t threshold);

void erodeImage(const uint8_t* d_input, uint8_t* d_output,
                size_t width, size_t height,
                int kernel_size = 3);

void dilateImage(const uint8_t* d_input, uint8_t* d_output,
                 size_t width, size_t height,
                 int kernel_size = 3);

void openingImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  int kernel_size = 3);

void closingImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  int kernel_size = 3);
