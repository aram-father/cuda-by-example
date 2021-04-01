/**
 * @file wave.cuh
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Wave
 * @version 0.1
 * @date 2021-04-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef __WAVE__CUH__
#define __WAVE__CUH__

#include <cstdint>

__global__ void kernel(std::uint8_t* p_image, std::uint32_t tick);

#endif