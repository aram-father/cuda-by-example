/**
 * @file julia_set.cuh
 * @author Wonseok Lee (aram_fahter@naver.com)
 * @brief Julia
 * @version 0.1
 * @date 2021-03-08
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef __JULIA_SET_CUH__
#define __JULIA_SET_CUH__

#include <iostream>
#include <cstdint>

class Complex
{
private:
    float real_;
    float imaginary_;
public:
    __device__ Complex(const float real, const float imaginary);
    
    __device__ Complex operator*(const Complex& rhs) const;
    __device__ Complex operator+(const Complex& rhs) const;

    __device__ float GetMagnitudeSquare(void) const;
    __device__ bool DoesJuliaExpansionConverge(const std::size_t length=200, const float threshold=1000.0) const;
};

__global__ void kernel(std::uint8_t* p_image);

#endif