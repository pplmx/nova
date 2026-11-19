#pragma once

#include <cstddef>
#include <cstdint>

void computeHistogram(const uint8_t* d_input, uint32_t* d_histogram, size_t width, size_t height, int bins = 256);

void computeHistogramPerChannel(const uint8_t* d_input, uint32_t* d_histogram_r, uint32_t* d_histogram_g, uint32_t* d_histogram_b, size_t width, size_t height);

void equalizeHistogram(const uint8_t* d_input, uint8_t* d_output, const uint32_t* d_histogram, size_t width, size_t height);
