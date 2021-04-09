cmake_minimum_required(VERSION 3.0.0)

project(05_inner_product LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

add_subdirectory(lib/inner_product)

add_executable(
  ${PROJECT_NAME} 05_inner_product.cu
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/inner_product
)
target_link_libraries(
  ${PROJECT_NAME} inner_product
)