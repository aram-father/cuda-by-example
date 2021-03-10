cmake_minimum_required(VERSION 3.0)

project(03_julia_set LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

add_subdirectory(lib/bitmap)
add_subdirectory(lib/julia_set)

add_executable(
    ${PROJECT_NAME} 03_julia_set.cu
)
target_include_directories(
    ${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitmap ${CMAKE_CURRENT_SOURCE_DIR}/lib/julia_set
)
target_link_libraries(
    ${PROJECT_NAME} bitmap julia_set
)