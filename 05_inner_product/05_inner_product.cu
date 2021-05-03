/**
 * @file 05_inner_product.cu
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Calculate inner product of two arrays
 * @version 0.1
 * @date 2021-04-08
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <ctime>
#include <cstdlib>

#include <chrono>
#include <memory>
#include <iostream>

#include "inner_product.cuh"

float* AllocateRandomFloatArray(const std::size_t length)
{
  float* ret = new float[length];
  float denominator = static_cast<float>(RAND_MAX);

  for (std::size_t idx = 0; idx < length; ++idx)
  {
    ret[idx] = std::rand() / denominator * (std::rand() % 2 ? -1 : 1);
  }

  return ret;
}

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cout << "Usage: ./05_inner_product length_of_arrays" << std::endl;
    exit(-1);
  }

  std::size_t length = static_cast<std::size_t>(std::atoi(argv[1]));
  
  std::shared_ptr<float> vector_a(AllocateRandomFloatArray(length));
  std::shared_ptr<float> vector_b(AllocateRandomFloatArray(length));

  auto cpu_s = std::chrono::system_clock::now();
  float cpu_result = InnerProductCPU(length, vector_a.get(), vector_b.get());
  auto cpu_e = std::chrono::system_clock::now();
  auto cpu_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_e - cpu_s);

  std::cout << "CPU Result: "<< cpu_result << std::endl;
  std::cout << "CPU Execution Time: "<< cpu_elapsed.count() << " msec" << std::endl;

  cudaError_t gpu_error;
  cudaEvent_t gpu_s, gpu_e;

  gpu_error = cudaEventCreate(&gpu_s);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }

  gpu_error = cudaEventCreate(&gpu_e);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }

  gpu_error = cudaEventRecord(gpu_s, 0);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }
  
  float gpu_result = InnerProductGPU(length, vector_a.get(), vector_b.get());
  
  gpu_error = cudaEventRecord(gpu_e, 0);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }

  gpu_error = cudaEventSynchronize(gpu_e);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }

  float gpu_elapsed;

  gpu_error = cudaEventElapsedTime(&gpu_elapsed, gpu_s, gpu_e);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }

  gpu_error = cudaEventDestroy(gpu_s);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }

  gpu_error = cudaEventDestroy(gpu_e);
  if (gpu_error)
  {
    std::cout << cudaGetErrorString(gpu_error) << std::endl;
    exit(-1);
  }
 
  std::cout << "GPU Result: "<< gpu_result << std::endl;
  std::cout << "GPU Execution Time: "<< gpu_elapsed << " msec" << std::endl;

  return 0;
}