cmake_minimum_required(VERSION 3.18)

project(CudaEfficientFeatures)

option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_TESTS   "Build tests" OFF)
set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")

add_subdirectory(modules)

if(BUILD_SAMPLES)
	add_subdirectory(samples)
endif()

if(BUILD_TESTS)
	add_subdirectory(tests)
endif()
