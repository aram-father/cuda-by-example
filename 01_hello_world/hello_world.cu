#include <iostream>
#include <cstdlib>

__global__ void kernel_add(int src_0, int src_1, int* dst)
{
  *dst = src_0 + src_1;
}

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cout << "Uasge: 01_hello_world number_0 number_1" << std::endl;
    exit(-1);
  }

  cudaError_t error;

  int src_0 = std::atoi(argv[1]);
  int src_1 = std::atoi(argv[2]);
  int *pdst, dst;

  error = cudaMalloc(reinterpret_cast<void**>(&pdst), sizeof(int));
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  kernel_add<<<1,1>>>(src_0, src_1, pdst);

  error = cudaMemcpy(&dst, pdst, sizeof(int), cudaMemcpyDeviceToHost);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  std::cout << src_0 << "+" << src_1 << "=" << dst << std::endl;

  cudaFree(pdst);

  return 0;
}