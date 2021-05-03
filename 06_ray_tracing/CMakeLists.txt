cmake_minimum_required(VERSION 3.0.0)

project(06_ray_tracing LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

add_subdirectory(lib/bitmap)
add_subdirectory(lib/sphere)
add_subdirectory(lib/ray_tracing)

add_executable(
  ${PROJECT_NAME}
  06_ray_tracing.cu
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitmap
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/sphere
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/ray_tracing
)
target_link_libraries(
  ${PROJECT_NAME}
  bitmap
  sphere
  ray_tracing
)
set_target_properties(
  ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)