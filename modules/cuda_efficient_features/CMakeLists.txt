cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME cuda_efficient_features)
set(INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INTERNAL_INCLUDE_DIR ${INTERFACE_INCLUDE_DIR})

# create project
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# dependent packages
find_package(OpenCV REQUIRED core features2d cudev cudafilters cudaimgproc cudawarping)
find_package(CUDAToolkit REQUIRED)

# target configuration
file(GLOB SRCS src/*.cpp src/*.cu src/*.h* ${INTERNAL_INCLUDE_DIR}/*.h*)
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE ${INTERNAL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} CUDA::cudart CUDA::cublas)
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${INTERFACE_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})

target_compile_options(${PROJECT_NAME} PRIVATE
$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-lineinfo -Xcompiler="\"/wd4819 /wd4244\"">
)

install(
	TARGETS ${PROJECT_NAME}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

install(
	DIRECTORY ${INTERFACE_INCLUDE_DIR}
	DESTINATION ${CMAKE_INSTALL_PREFIX}
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
