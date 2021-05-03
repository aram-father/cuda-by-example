/**
 * @file ray_tracing.cuh
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Header for ray tracing
 * @version 0.1
 * @date 2021-05-03
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef __RAY_TRACING_CUH__
#define __RAY_TRACING_CUH__

#include <cstdint>

#include "sphere.cuh"

extern __constant__ Sphere kSpheresDev[1024];

__global__ void DoRayTracingUsingGlobalMemory(
  const std::uint32_t number_of_spheres,
  const Sphere* const p_spheres,
  std::uint8_t* const p_image);

__global__ void DoRayTracingUsingConstantMemory(
  const std::uint32_t number_of_spheres,
  std::uint8_t* const p_image);

#endif