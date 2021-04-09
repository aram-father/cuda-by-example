/**
 * @file inner_product.cu
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Source for inner product
 * @version 0.1
 * @date 2021-04-08
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "inner_product.cuh"

#define MAX_SM 16
#define MAX_THREAD 1024

float InnerProductCPU(
  const std::size_t length,
  const float* const vector_a,
  const float* const vector_b)
{
  float ret = 0.0f;

  for (std::size_t idx = 0; idx < length; ++idx)
  {
    ret += vector_a[idx] * vector_b[idx];
  }

  return ret;
}

float InnerProductGPU(
  const std::size_t length,
  const float* const vector_a,
  const float* const vector_b)
{
  const std::size_t blocks_per_grid = MAX_SM;
  const std::size_t threads_per_block = MAX_THREAD;

  dim3 grid_dim(blocks_per_grid, 1, 1);
  dim3 block_dim(threads_per_block, 1, 1);

  float *dev_vector_a = nullptr;
  float *dev_vector_b = nullptr;
  float *dev_reduced = nullptr;

  float *host_reduced = new float[blocks_per_grid];

  cudaError_t error;
  
  error = cudaMalloc(reinterpret_cast<void**>(&dev_vector_a), sizeof(float) * length);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  error = cudaMalloc(reinterpret_cast<void**>(&dev_vector_b), sizeof(float) * length);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  error = cudaMalloc(reinterpret_cast<void**>(&dev_reduced), sizeof(float) * blocks_per_grid);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  error = cudaMemcpy(dev_vector_a, vector_a, sizeof(float) * length, cudaMemcpyHostToDevice);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  error = cudaMemcpy(dev_vector_b, vector_b, sizeof(float) * length, cudaMemcpyHostToDevice);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  InnerProductKernel<<<grid_dim, block_dim>>>(length, dev_vector_a, dev_vector_b, dev_reduced);

  error = cudaMemcpy(host_reduced, dev_reduced, sizeof(float) * blocks_per_grid, cudaMemcpyDeviceToHost);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  error = cudaFree(dev_vector_a);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  error = cudaFree(dev_vector_b);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  error = cudaFree(dev_reduced);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  float ret = 0.0f;

  for (std::size_t idx = 0; idx < blocks_per_grid; ++idx)
  {
    ret += host_reduced[idx];
  }

  free(host_reduced);

  return ret;
}

__global__ void InnerProductKernel(
  const std::size_t length,
  const float* const vector_a,
  const float* const vector_b,
  float* const reduced)
{
  __shared__ float cache[MAX_THREAD];

  const int stride = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float temp = 0.0f;

  while (idx < length)
  {
    temp += vector_a[idx] * vector_b[idx];
    idx += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();

  int reduction_size = 2;
  while (reduction_size <= blockDim.x)
  {
    if (threadIdx.x % reduction_size == 0)
    {
      cache[threadIdx.x] += cache[threadIdx.x + reduction_size / 2];
    }

    __syncthreads();

    reduction_size *= 2;
  }

  reduced[blockIdx.x] = cache[0];
}