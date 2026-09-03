# =============================================================================
# Nova Examples (TASK-064 re-integration)
# =============================================================================
# The examples demonstrate the library's public API as standalone executables.
# They were broken for a long time: they referenced the pre-v2 nova::* API and
# were not built at all despite examples/README.md claiming otherwise (ISS-002 /
# ISS-003). Each is ported to the current cuda::* API as it lands here, so a
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
