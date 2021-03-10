/**
 * @file main.cc
 * @author your name (you@domain.com)
 * @brief Main
 * @version 0.1
 * @date 2021-03-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <cstdlib>
#include <iostream>

#include "bitmap.hpp"
#include "julia_set.cuh"

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cout << "Usage: ./main dim_x dim_y" << std::endl;
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

  dim3 grid(nrow, ncol);
  kernel<<<grid, 1>>>(p_image_dev);

  error = cudaMemcpy(bmp.image().data, p_image_dev, nrow * ncol * 4, cudaMemcpyDeviceToHost);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
  }

  error = cudaFree(p_image_dev);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  bmp.ShowAndWait("test");

  return 0;
}