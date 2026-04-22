#pragma once

#include <cstddef>
#include <cstdint>

void gaussianBlur(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  float sigma = 1.0f, int kernel_size = 5);
