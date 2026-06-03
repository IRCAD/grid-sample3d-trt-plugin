cmake_minimum_required(VERSION 3.10)

project(grid_sample_3d_plugin LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS "${CUDAToolkit_INCLUDE_DIRS}" PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(
NVINFER_LIBRARY nvinfer_10 nvinfer PATHS "${CUDAToolkit_LIBRARY_DIR};${CMAKE_BINARY_DIR}" PATH_SUFFIXES lib
lib64
)

file(GLOB SOURCES "./src/*.cpp")
file(GLOB CU_SOURCE "./src/*.cu")

message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CU_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE "./src" ${CUDAToolkit_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${NVINFER_LIBRARY})

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

