cmake_minimum_required(VERSION 3.0.0)

project(02_device_query LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

add_executable(
    ${PROJECT_NAME} device_query.cu
)
target_include_directories(
    ${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)