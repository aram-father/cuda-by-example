/**
 * @file 04_wave.cu
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Wave
 * @version 0.1
 * @date 2021-04-01
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "bitmap.hpp"
#include "wave.cuh"

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cout << "Usage: ./04_wave dim_x dim_y" << std::endl;
    exit(-1);
  }

  int nrow = std::atoi(argv[1]);
  int ncol = std::atoi(argv[2]);

  std::uint8_t* p_image_dev;
  Bitmap bmp(nrow, ncol);

  cudaError_t error;
  error = cudaMalloc(reinterpret_cast<void**>(&p_image_dev), nrow * ncol * 4);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  std::uint32_t tick = 0;
  while (bmp.ShowAndWait("test", 50) == -1)
  {
    dim3 grid_dim(nrow/16, ncol/16);
    dim3 block_dim(16, 16);

    kernel<<<grid_dim, block_dim>>>(p_image_dev, tick);

    error = cudaMemcpy(bmp.image().data, p_image_dev, nrow * ncol * 4, cudaMemcpyDeviceToHost);
    if (error)
    {
      std::cout << cudaGetErrorString(error) << std::endl;
    }

    tick += 50;
  }

  error = cudaFree(p_image_dev);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  return 0;
}