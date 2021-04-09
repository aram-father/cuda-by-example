/**
 * @file inner_product.cuh
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Header for inner product
 * @version 0.1
 * @date 2021-04-08
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef __INNER_PRODUCT_CUH__
#define __INNER_PRODUCT_CUH__

#include <iostream>

float InnerProductCPU(
  const std::size_t length,
  const float* const vector_a,
  const float* const vector_b);

float InnerProductGPU(
  const std::size_t length,
  const float* const vector_a,
  const float* const vector_b);

__global__ void InnerProductKernel(
  const std::size_t length,
  const float* const vector_a,
  const float* const vector_b,
  float* const reduced);

#endif