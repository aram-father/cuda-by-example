cmake_minimum_required(VERSION 3.0.0)

project(04_wave LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

add_subdirectory(lib/bitmap)
add_subdirectory(lib/wave)

add_executable(
    ${PROJECT_NAME} 04_wave.cu
)
target_include_directories(
    ${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitmap ${CMAKE_CURRENT_SOURCE_DIR}/lib/wave
)
target_link_libraries(
    ${PROJECT_NAME} bitmap wave
)