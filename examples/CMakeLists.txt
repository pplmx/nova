# =============================================================================
# Nova Examples (re-integration)
# =============================================================================
# The examples demonstrate the library's public API as standalone executables.
# They were broken for a long time: they referenced the pre-v2 nova::* API and
# were not built at all despite examples/README.md claiming otherwise. Each is
# ported to the current cuda::* API as it lands here, so a
# target exists only for a ported example — `cmake --build build --target <ex>`
# builds one, `--target neural_net image_processing graph_algorithms ...` all.

add_executable(neural_net neural_net.cpp)
target_link_libraries(neural_net PRIVATE cuda_impl CUDA::cublasLt)
target_include_directories(neural_net PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(image_processing image_processing.cpp)
target_link_libraries(image_processing PRIVATE cuda_impl)
target_include_directories(image_processing PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(graph_algorithms graph_algorithms.cpp)
target_link_libraries(graph_algorithms PRIVATE cuda_impl)
target_include_directories(graph_algorithms PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(distributed_training distributed_training.cpp)
target_link_libraries(distributed_training PRIVATE cuda_impl)
target_include_directories(distributed_training PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Register the examples as tests. Nothing ran them before: they were built by the
# default build and then never executed, so a change that broke one (an API rename,
# a wrong kernel argument) showed up only when a human tried the demo. The image
# example needs an input file, so it is driven through its own --generate first;
# the distributed one needs NCCL and is registered only where the library has it.
add_test(NAME example_neural_net COMMAND neural_net --batch 8 --seq_len 4 --hidden 16)
add_test(NAME example_graph_algorithms COMMAND graph_algorithms --algorithm bfs --nodes 256 --edges 1024)

# The image example is three kernels over a generated pattern, one test each, in a
# scratch directory so parallel runs cannot collide.
set(NOVA_EXAMPLE_IMAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/example_image")
file(MAKE_DIRECTORY "${NOVA_EXAMPLE_IMAGE_DIR}")
# Each kernel is checked for a second thing beyond its exit status: that the output
# is not byte-identical to the input. A kernel that returns success without writing
# a pixel used to pass, which is the failure mode an example is supposed to make
# visible.
find_program(NOVA_EXAMPLE_PYTHON NAMES python3 python)
foreach(example_kernel sobel blur morphology)
    add_test(NAME example_image_${example_kernel}
             COMMAND image_processing --input ${NOVA_EXAMPLE_IMAGE_DIR}/in.pgm
                     --output ${NOVA_EXAMPLE_IMAGE_DIR}/out_${example_kernel}.pgm
                     --kernel ${example_kernel})
    set_tests_properties(example_image_${example_kernel} PROPERTIES
        FIXTURES_REQUIRED example_image_input
        FIXTURES_SETUP "example_image_${example_kernel}_output")

    if(NOVA_EXAMPLE_PYTHON)
        add_test(NAME example_image_${example_kernel}_transforms
                 COMMAND ${NOVA_EXAMPLE_PYTHON}
                         ${CMAKE_SOURCE_DIR}/scripts/examples/check_transformed.py
                         ${NOVA_EXAMPLE_IMAGE_DIR}/in.pgm
                         ${NOVA_EXAMPLE_IMAGE_DIR}/out_${example_kernel}.pgm)
        # Depends on the file the kernel test wrote.
        set_tests_properties(example_image_${example_kernel}_transforms PROPERTIES
            FIXTURES_REQUIRED "example_image_input;example_image_${example_kernel}_output")
    else()
        message(STATUS "Nova: python3 not found; example image outputs are not content-checked")
    endif()
endforeach()

add_test(NAME example_image_generate
         COMMAND image_processing --generate 64x48 --input ${NOVA_EXAMPLE_IMAGE_DIR}/in.pgm)
set_tests_properties(example_image_generate PROPERTIES FIXTURES_SETUP example_image_input)

# NCCL is what the example actually exercises; without it the all-reduce fails with
# "NCCL not enabled" (exit 1), which is correct behaviour but not a test of the demo.
if(NOVA_ENABLE_NCCL AND TARGET cuda_nccl)
    get_target_property(_example_nccl_defs cuda_nccl INTERFACE_COMPILE_DEFINITIONS)
    if(_example_nccl_defs MATCHES "NOVA_NCCL_ENABLED=1")
        add_test(NAME example_distributed_training COMMAND distributed_training --chunk 256)
    endif()
endif()
