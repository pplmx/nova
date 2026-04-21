#pragma once

#include <cstddef>
#include <cstring>
#include <cmath>
#include <cstdlib>

static inline void generateCheckerboard(unsigned char* buffer, size_t width, size_t height, size_t cellSize) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            bool cellX = (x / cellSize) % 2 == 0;
            bool cellY = (y / cellSize) % 2 == 0;
            unsigned char value = (cellX == cellY) ? 255 : 0;
            size_t idx = (y * width + x) * 3;
            buffer[idx] = buffer[idx + 1] = buffer[idx + 2] = value;
        }
    }
}

static inline void generateGradient(unsigned char* buffer, size_t width, size_t height) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 3;
            buffer[idx] = static_cast<unsigned char>((x * 255) / width);
            buffer[idx + 1] = static_cast<unsigned char>((y * 255) / height);
            buffer[idx + 2] = static_cast<unsigned char>(((x + y) * 255) / (width + height));
        }
    }
}

static inline void generateSolid(unsigned char* buffer, size_t width, size_t height, unsigned char value) {
    std::memset(buffer, value, width * height * 3);
}

static inline bool compareBuffers(const unsigned char* a, const unsigned char* b, size_t size, float tolerance = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (diff > tolerance * 255.0f) {
            return false;
        }
    }
    return true;
}
