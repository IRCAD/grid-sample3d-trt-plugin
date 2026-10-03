cmake_minimum_required(VERSION 3.10)

set(PARENT_PROJECT_NAME ${PROJECT_NAME})

project(test LANGUAGES CXX CUDA)
enable_language(CUDA)
find_package(CUDAToolkit)

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(TEST_GRID_SAMPLE test_grid_sample)

add_executable(${TEST_GRID_SAMPLE} test.cpp)

target_include_directories(${TEST_GRID_SAMPLE} PUBLIC ${PROJECT_INCLUDE_DIR} ${CUDA_ROOT}/include)
target_link_libraries(${TEST_GRID_SAMPLE} ${PARENT_PROJECT_NAME} CUDA::cudart)

# CMAKE_CUDA_ARCHITECTURES is inherited from the parent scope (defaults to
# "native", or whatever -DCMAKE_CUDA_ARCHITECTURES=... was passed on the
# command line), so no per-target override is needed here.

# test.cpp reads data files using paths relative to the top-level build
# directory (e.g. "../test/data/input.txt"), not this target's own binary
# directory, so the working directory must be set explicitly.
add_test(NAME ${TEST_GRID_SAMPLE}
         COMMAND ${TEST_GRID_SAMPLE}
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})