cmake_minimum_required(VERSION 3.18)

project(samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

set(SRCS_COMMON sample_common.cpp sample_common.h)

add_executable(sample_feature_extraction sample_feature_extraction.cpp ${SRCS_COMMON})
target_link_libraries(sample_feature_extraction ${OpenCV_LIBS} cuda_efficient_features)

add_executable(sample_feature_matching sample_feature_matching.cpp ${SRCS_COMMON})
target_link_libraries(sample_feature_matching ${OpenCV_LIBS} cuda_efficient_features)

add_executable(sample_image_sequence sample_image_sequence.cpp ${SRCS_COMMON})
target_link_libraries(sample_image_sequence ${OpenCV_LIBS} cuda_efficient_features)

add_executable(sample_benchmark sample_benchmark.cpp ${SRCS_COMMON})
target_link_libraries(sample_benchmark ${OpenCV_LIBS} efficient_features cuda_efficient_features)

add_executable(hpatches_description hpatches_description.cpp ${SRCS_COMMON})
target_link_libraries(hpatches_description ${OpenCV_LIBS} efficient_features cuda_efficient_features OpenMP::OpenMP_CXX)
