cmake_minimum_required(VERSION 3.18)

project(tests LANGUAGES CXX)

# required packages
find_package(OpenCV REQUIRED)

if (MSVC)
	option(gtest_force_shared_crt "Force Gmock to use standard compiler flags" ON)
endif()

add_subdirectory(googletest)

file(GLOB SRCS ./*.cpp ./*.h*)
cuda_add_executable(tests ${SRCS})

target_compile_features(tests PRIVATE cxx_std_17)
target_compile_options(
	tests PRIVATE
	$<$<CXX_COMPILER_ID:GCC>:-O3 -Wall>
	$<$<CXX_COMPILER_ID:Clang>:-O3 -Wall>
	$<$<CXX_COMPILER_ID:MSVC>:/wd4819>
)
target_include_directories(tests PRIVATE ${gtest_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tests gtest ${OpenCV_LIBS} efficient_features cuda_efficient_features)
target_compile_definitions(tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
