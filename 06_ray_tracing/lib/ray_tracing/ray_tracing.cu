/**
 * @file ray_tracing.cu
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Source for ray tracing
 * @version 0.1
 * @date 2021-05-03
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "ray_tracing.cuh"

__constant__ Sphere kSpheresDev[1024];

__global__ void DoRayTracingUsingGlobalMemory(
  const std::uint32_t number_of_spheres,
  const Sphere* const p_spheres,
  std::uint8_t* const p_image)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = y * blockDim.x * gridDim.x + x;

  float ray_x = x - MAX_X;
  float ray_y = y - MAX_Y;

  float r = 0.0f, g = 0.0f, b = 0.0f;
  float max_depth = -INF;
  for (int sphere_idx = 0; sphere_idx < number_of_spheres; ++sphere_idx)
  {
    float gradation;
    float depth = p_spheres[sphere_idx].GetDepth(ray_x, ray_y, &gradation);
    if (depth > max_depth)
    {
      r = p_spheres[sphere_idx].channel_r_ * gradation;
      g = p_spheres[sphere_idx].channel_g_ * gradation;
      b = p_spheres[sphere_idx].channel_b_ * gradation;
      max_depth = depth;
    }
  }

  p_image[4 * offset + 0] = (std::uint8_t)(r * 255);
  p_image[4 * offset + 1] = (std::uint8_t)(g * 255);
  p_image[4 * offset + 2] = (std::uint8_t)(b * 255);
  p_image[4 * offset + 3] = 255;
}

__global__ void DoRayTracingUsingConstantMemory(
  const std::uint32_t number_of_spheres,
  std::uint8_t* const p_image)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = y * blockDim.x * gridDim.x + x;

  float ray_x = x - MAX_X;
  float ray_y = y - MAX_Y;

  float r = 0.0f, g = 0.0f, b = 0.0f;
  float max_depth = -INF;
  for (int sphere_idx = 0; sphere_idx < number_of_spheres; ++sphere_idx)
  {
    float gradation;
    float depth = kSpheresDev[sphere_idx].GetDepth(ray_x, ray_y, &gradation);
    if (depth > max_depth)
    {
      r = kSpheresDev[sphere_idx].channel_r_ * gradation;
      g = kSpheresDev[sphere_idx].channel_g_ * gradation;
      b = kSpheresDev[sphere_idx].channel_b_ * gradation;
      max_depth = depth;
    }
  }

  p_image[4 * offset + 0] = (std::uint8_t)(r * 255);
  p_image[4 * offset + 1] = (std::uint8_t)(g * 255);
  p_image[4 * offset + 2] = (std::uint8_t)(b * 255);
  p_image[4 * offset + 3] = 255;  
}