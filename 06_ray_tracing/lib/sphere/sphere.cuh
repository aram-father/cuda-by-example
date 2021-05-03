/**
 * @file sphere.cuh
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Header for sphere
 * @version 0.1
 * @date 2021-05-03
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef __SPHERE_CUH__
#define __SPHERE_CUH__

#include <limits>

#define MAX_X ((float)(512.0f))
#define MAX_Y ((float)(512.0f))
#define MAX_Z ((float)(512.0f))
#define INF ((float)(2e10f))

struct Sphere
{
public:
  float radius_;
  
  float center_x_;
  float center_y_;
  float center_z_;

  float channel_r_;
  float channel_g_;
  float channel_b_;

public:
  void Initialize(void);
  __device__ float GetDepth(const float ray_x, const float ray_y, float* const gradation) const;
};

#endif