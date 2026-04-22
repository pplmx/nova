find_package(CUDAToolkit REQUIRED)

add_executable(test_patterns-tests
    test_patterns_test.cpp
)
target_link_libraries(test_patterns-tests PRIVATE GTest::gtest_main)
target_include_directories(test_patterns-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/data
)
include(GoogleTest)
gtest_discover_tests(test_patterns-tests)

add_executable(cuda-samples-tests
    image_utils_test.cu
    gaussian_blur_test.cu
    sobel_edge_test.cu
    brightness_test.cu
    reduce_test.cu
    scan_test.cu
    sort_test.cu
    matrix_mult_test.cu
)

set(CUDA_KERNEL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/kernel)
set(CUDA_ALGO_DIR ${CMAKE_SOURCE_DIR}/include/cuda/algo)
set(CUDA_API_DIR ${CMAKE_SOURCE_DIR}/include/cuda/api)

target_link_libraries(cuda-samples-tests
    PRIVATE
    GTest::gtest_main
    GTest::gmock
    CUDA::cudart
    cuda_api
    cuda_impl
)
target_include_directories(cuda-samples-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data
    ${CUDA_KERNEL_DIR}
    ${CUDA_ALGO_DIR}
    ${CUDA_API_DIR}
)

target_compile_options(cuda-samples-tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -lineinfo
    >
)

include(GoogleTest)
gtest_discover_tests(cuda-samples-tests)
