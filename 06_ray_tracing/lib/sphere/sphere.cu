/**
 * @file sphere.cu
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Source for sphere
 * @version 0.1
 * @date 2021-05-03
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "sphere.cuh"

#include <cstdlib>

void Sphere::Initialize(void)
{ 
  radius_ = std::rand() * 100.0f / RAND_MAX + 20.0f;

  center_x_ = std::rand() * 2.0f * MAX_X / RAND_MAX - MAX_X;
  center_y_ = std::rand() * 2.0f * MAX_Y / RAND_MAX - MAX_Y;
  center_z_ = std::rand() * 2.0f * MAX_Z / RAND_MAX - MAX_Z;

  channel_r_ = std::rand() * 1.0f / RAND_MAX;
  channel_g_ = std::rand() * 1.0f / RAND_MAX;
  channel_b_ = std::rand() * 1.0f / RAND_MAX;
}

__device__ float Sphere::GetDepth(const float ray_x, const float ray_y, float* const gradation) const
{
  float dx = ray_x - center_x_;
  float dy = ray_y - center_y_;

  if (dx * dx + dy * dy < radius_ * radius_)
  {
    float dz = sqrtf(radius_ * radius_ - dx * dx - dy * dy);
    *gradation = dz / sqrtf(radius_ * radius_);
    return center_z_ + dz;
  }
  else
  {
    return -INF;
  }
}