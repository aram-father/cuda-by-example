cmake_minimum_required(VERSION 3.0.0)

project(sphere LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

add_library(
  ${PROJECT_NAME} sphere.cu
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
  ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)