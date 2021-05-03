cmake_minimum_required(VERSION 3.0.0)

project(ray_tracing LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

add_library(
  ${PROJECT_NAME} ray_tracing.cu
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../sphere
)
set_target_properties(
  ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)