cmake_minimum_required(VERSION 3.10)

project(grid_sample_3d_plugin LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Allow -DCMAKE_CUDA_ARCHITECTURES=... to override, but otherwise default to
# building for whatever GPU(s) are present on the machine doing the build
# (CMake >= 3.24) instead of a single hardcoded architecture (e.g. "89",
# which only matches Ada-generation GPUs and silently produces a binary that
# can't run/load on other hardware).
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Building the standalone C++ test executable (test/) is independent of the
# TensorRT discovery below only in the sense that it doesn't need TensorRT
# headers itself, but it does link against the plugin library target which
# always requires TensorRT to build. Disable with -DBUILD_TESTS=OFF if only
# the plugin shared library is needed (e.g. packaging for a Python
# consumer that just loads libgrid_sample_3d_plugin.so via ctypes).
option(BUILD_TESTS "Build the standalone C++ test executable" ON)

# -----------------------------------------------------------------------------
# TensorRT discovery
#
# TensorRT is not part of the CUDA toolkit, so its headers/libraries can live
# in many different places depending on how it was installed: an apt package
# (/usr/include/x86_64-linux-gnu, /usr/lib/x86_64-linux-gnu), a manually
# downloaded SDK tarball, or the pip `tensorrt`/`tensorrt-cu12` wheels used by
# python venvs/conda envs (which ship libnvinfer*.so under
# <env>/lib/pythonX.Y/site-packages/tensorrt_libs but no headers at all).
#
# Rather than guessing a single location, allow explicit overrides and fall
# back to searching common locations (including the active Python
# environment's tensorrt_libs, if any):
#
#   cmake -B build \
#     -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT-X.Y.Z/include \
#     -DTENSORRT_LIB_DIR=/path/to/dir/containing/libnvinfer.so
# -----------------------------------------------------------------------------

set(TENSORRT_LIB_DIR "" CACHE PATH "Directory containing libnvinfer.so (optional override)")

find_package(Python3 QUIET COMPONENTS Interpreter)
set(_TENSORRT_PIP_LIB_HINT "")
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))"
    OUTPUT_VARIABLE _TENSORRT_PIP_LIB_HINT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

# NOTE: TENSORRT_INCLUDE_DIR is intentionally *not* pre-declared with
# `set(... CACHE ...)` here. find_path() only performs a search when the
# cache variable does not already exist; pre-declaring it (even to "") would
# make find_path() skip the search unconditionally. Passing
# -DTENSORRT_INCLUDE_DIR=/some/path on the cmake command line still works as
# an override, since that populates the cache before this script runs.
find_path(
  TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS "${CUDAToolkit_INCLUDE_DIRS}" /usr/include/x86_64-linux-gnu /usr/include
)

# find_library()'s NAMES only apply the standard lib<name>.so
# prefix/suffix convention, so a literal versioned filename like
# "libnvinfer.so.11" in NAMES never actually matches anything. This matters
# in practice: pip TensorRT wheels (tensorrt/tensorrt-cu12) ship only the
# versioned libnvinfer.so.<N> file with no unversioned "libnvinfer.so"
# symlink, so a plain find_library() search of that directory always fails
# and silently falls through to whatever "libnvinfer.so" happens to exist on
# the system (e.g. a different major version installed via apt) -- exactly
# the kind of mismatch that causes ONNX Runtime / TensorRT version confusion.
# So: explicitly look for both forms in the user-provided/auto-detected
# directories first, and only fall back to a conventional system search.
set(_TENSORRT_LIB_SEARCH_DIRS "")
if(TENSORRT_LIB_DIR)
  list(APPEND _TENSORRT_LIB_SEARCH_DIRS "${TENSORRT_LIB_DIR}")
endif()
if(_TENSORRT_PIP_LIB_HINT)
  list(APPEND _TENSORRT_LIB_SEARCH_DIRS "${_TENSORRT_PIP_LIB_HINT}")
endif()

set(NVINFER_LIBRARY "")
foreach(_dir ${_TENSORRT_LIB_SEARCH_DIRS})
  if(NOT NVINFER_LIBRARY)
    if(EXISTS "${_dir}/libnvinfer.so")
      set(NVINFER_LIBRARY "${_dir}/libnvinfer.so")
    else()
      file(GLOB _nvinfer_versioned_candidates "${_dir}/libnvinfer.so.*")
      if(_nvinfer_versioned_candidates)
        list(GET _nvinfer_versioned_candidates 0 NVINFER_LIBRARY)
      endif()
    endif()
  endif()
endforeach()

if(NOT NVINFER_LIBRARY)
  find_library(
    NVINFER_LIBRARY
    NAMES nvinfer
    PATHS "${CUDAToolkit_LIBRARY_DIR}" /usr/lib/x86_64-linux-gnu /usr/lib
    PATH_SUFFIXES lib lib64
  )
endif()

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR
    "Could not find NvInfer.h. Set -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include "
    "to the directory containing NvInfer.h (e.g. a downloaded TensorRT SDK's include/ "
    "directory, or /usr/include/x86_64-linux-gnu if TensorRT was installed via apt).")
endif()

if(NOT NVINFER_LIBRARY)
  message(FATAL_ERROR
    "Could not find libnvinfer. Set -DTENSORRT_LIB_DIR=/path/to/dir to the directory "
    "containing libnvinfer.so[.N] (e.g. a conda env's "
    "lib/pythonX.Y/site-packages/tensorrt_libs, or /usr/lib/x86_64-linux-gnu if "
    "TensorRT was installed via apt).")
endif()

message(STATUS "Using TensorRT headers from: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Using TensorRT library: ${NVINFER_LIBRARY}")

file(GLOB SOURCES "./src/*.cpp")
file(GLOB CU_SOURCE "./src/*.cu")

message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CU_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE "./src" ${CUDAToolkit_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${NVINFER_LIBRARY})

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

