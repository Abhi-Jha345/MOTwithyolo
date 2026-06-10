cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(tank_tracker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT
set(TRT_INCLUDE /usr/include/aarch64-linux-gnu)
set(TRT_LIB     /usr/lib/aarch64-linux-gnu)
include_directories(${TRT_INCLUDE})

# Sources
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/tracker.cpp
    src/preprocess.cu
)

add_executable(tank_tracker ${SOURCES})

target_include_directories(tank_tracker PRIVATE src)

target_link_libraries(tank_tracker
    ${OpenCV_LIBS}
    ${TRT_LIB}/libnvinfer.so
    ${TRT_LIB}/libnvinfer_plugin.so
    ${CUDA_LIBRARIES}
    cuda
    cudart
)

set_target_properties(tank_tracker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "87"   # Jetson Orin = sm_87
)
