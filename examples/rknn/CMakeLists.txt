cmake_minimum_required(VERSION 3.6)

project(rknn)

# simple
set(TEST_TARGET simple)
add_executable(${TEST_TARGET} simple.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

# matrix_mul
set(TEST_TARGET matrix_mul)
add_executable(${TEST_TARGET} matrix_mul.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml m)
