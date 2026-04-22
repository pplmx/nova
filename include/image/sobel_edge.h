#pragma once

#include <cstddef>
#include <cstdint>

void sobelEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                        size_t width, size_t height,
                        float threshold = 30.0f);
