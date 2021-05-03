/**
 * @file 06_ray_tracing.cu
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Main for ray tracing example
 * @version 0.1
 * @date 2021-05-03
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <ctime>
#include <string>
#include <cstdlib>
#include <iostream>

#include "bitmap.hpp"
#include "sphere.cuh"
#include "ray_tracing.cuh"

static void HandleError(cudaError_t error, const char *file, int line)
{
  if (error != cudaSuccess)
  {
    std::cout << cudaGetErrorString(error) << " @ " << line << " of " << file << std::endl;
    exit(-1);
  }
}

#define HANDLE_ERROR(error) (HandleError((error), __FILE__, __LINE__))

int main(int argc, char **argv)
{
  int number_of_spheres;

  try
  {
    if (argc != 2 || !(number_of_spheres = std::stoi(argv[1])) || number_of_spheres >= 1024)
    {
      throw std::string("Usage: ./06_ray_tracing NUMBER_OF_SPHERES (0,1024]");
    }
  }
  catch (std::string usage_exception)
  {
    std::cout << usage_exception << std::endl;
    exit(-1);
  }

  Sphere* p_spheres_host = new Sphere[number_of_spheres];
  if (!p_spheres_host)
  {
    std::cout << "host heap allocation failed" << std::endl;
    exit(-1);
  }

  std::srand(std::time(0));
  for (int sphere_idx = 0; sphere_idx < number_of_spheres; ++sphere_idx)
  {
    p_spheres_host[sphere_idx].Initialize();
  }

  float elapsed_time;
  cudaEvent_t start, end;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&end));

  const std::uint32_t kNRow = MAX_X * 2;
  const std::uint32_t kNCol = MAX_Y * 2;
  
  Bitmap bmp(kNRow, kNCol);
  
  Sphere* p_spheres_dev;
  std::uint8_t* p_image_dev;

  dim3 grid_dim(kNRow / 16, kNCol / 16, 1);
  dim3 block_dim(16, 16, 1);

  // Global memory version
  HANDLE_ERROR(cudaEventRecord(start, 0));
  
  HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&p_spheres_dev), sizeof(Sphere) * number_of_spheres));
  HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&p_image_dev), sizeof(std::uint8_t) * kNRow * kNCol * 4));
  HANDLE_ERROR(cudaMemcpy(p_spheres_dev, p_spheres_host, sizeof(Sphere) * number_of_spheres, cudaMemcpyHostToDevice));

  DoRayTracingUsingGlobalMemory<<<grid_dim, block_dim>>>(number_of_spheres, p_spheres_dev, p_image_dev);

  HANDLE_ERROR(cudaMemcpy(bmp.image().data, p_image_dev, sizeof(std::uint8_t) * kNRow * kNCol * 4, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(p_spheres_dev));
  HANDLE_ERROR(cudaFree(p_image_dev));
  
  HANDLE_ERROR(cudaEventRecord(end, 0));
  HANDLE_ERROR(cudaEventSynchronize(end));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, end));
  std::cout << "Global memory version elapsed time: " << elapsed_time << std::endl;
  bmp.ShowAndWait();

  // Constant memory version
  HANDLE_ERROR(cudaEventRecord(start, 0));

  HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&p_image_dev), sizeof(std::uint8_t) * kNRow * kNCol * 4));
  HANDLE_ERROR(cudaMemcpyToSymbol(kSpheresDev, p_spheres_host, sizeof(Sphere) * number_of_spheres));

  DoRayTracingUsingConstantMemory<<<grid_dim, block_dim>>>(number_of_spheres, p_image_dev);

  HANDLE_ERROR(cudaMemcpy(bmp.image().data, p_image_dev, sizeof(std::uint8_t) * kNRow * kNCol * 4, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(p_image_dev));
  
  HANDLE_ERROR(cudaEventRecord(end, 0));
  HANDLE_ERROR(cudaEventSynchronize(end));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, end));
  std::cout << "Constant memory version elapsed time: " << elapsed_time << std::endl;
  bmp.ShowAndWait();

  // Common termination
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(end));

  free(p_spheres_host);

  return 0;
}