cmake_minimum_required(VERSION 3.20)
project(ichol LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_subdirectory(src)
add_subdirectory(test)
