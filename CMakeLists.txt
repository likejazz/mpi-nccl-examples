cmake_minimum_required(VERSION 2.8)

project(nccl-examples)

SET(NCCL_ROOT_DIR "/usr/local/cuda/")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

# ---[ Flags
if (UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++11")
endif ()

# ---[ NCCL
find_package(NCCL REQUIRED)
if (NCCL_FOUND)
    add_definitions(-DUSE_NCCL)
    include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
endif ()

include(FindCUDA)
include_directories(/usr/local/cuda/include ${MPI_INCLUDE_PATH})

add_subdirectory(src)
