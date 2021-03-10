cmake_minimum_required(VERSION 3.0.0)

project(bitmap)
add_compile_options(-std=c++11 -Wall)

find_package(OpenCV REQUIRED)

add_library(
    ${PROJECT_NAME} bitmap.cc
)
target_include_directories(
    ${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    ${PROJECT_NAME} ${OpenCV_LIBRARIES}
)