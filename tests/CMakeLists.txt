find_package(CUDAToolkit REQUIRED)

add_executable(cuda-samples-tests
    image_utils_test.cu
    gaussian_blur_test.cu
    sobel_edge_test.cu
    brightness_test.cu
)

target_link_libraries(cuda-samples-tests
    PRIVATE
    GTest::gtest_main
    GTest::gmock
    CUDA::cudart
    ${PROJECT_NAME}_lib
)

target_include_directories(cuda-samples-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data
)

target_compile_options(cuda-samples-tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -lineinfo
    >
)

include(GoogleTest)
gtest_discover_tests(cuda-samples-tests)
