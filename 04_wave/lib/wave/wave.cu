/**
 * @file wave.cu
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Wave
 * @version 0.1
 * @date 2021-04-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <iostream>

#include "wave.cuh"

__global__ void kernel(std::uint8_t* p_image, std::uint32_t tick)
{
  const int kGridX = threadIdx.x + blockIdx.x * blockDim.x;
  const int kGridY = threadIdx.y + blockIdx.y * blockDim.y;

  const int kOffset = kGridY * (gridDim.x * blockDim.x) + kGridX;

  const int kImgH = gridDim.y * blockDim.y;
  const int kImgW = gridDim.x * blockDim.x;
  const float kImgX = kGridX - kImgW / 2;
  const float kImgY = kGridY - kImgH / 2;

  const float kDist = sqrtf(kImgX * kImgX + kImgY * kImgY);

  const unsigned char kGrey = (unsigned char)(128.0f + 127.0f*cos(kDist/10.0f - tick/7.0f) / (kDist/10.0f + 1.0f));

  p_image[4*kOffset+0] = kGrey;
  p_image[4*kOffset+1] = kGrey;
  p_image[4*kOffset+2] = kGrey;
  p_image[4*kOffset+3] = 255;
}