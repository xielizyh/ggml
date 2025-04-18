cmake_minimum_required(VERSION 3.6)

project(rknn)

set(TEST_TARGET matrix_mul)
add_executable(${TEST_TARGET} matrix_mul.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
target_link_libraries(${TEST_TARGET} PRIVATE ${RKNN_RT_LIB})
