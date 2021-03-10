/**
 * @file julia_set.cu
 * @author Wonseok Lee (aram_fahter@naver.com)
 * @brief Julia
 * @version 0.1
 * @date 2021-03-08
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "julia_set.cuh"

__device__ Complex::Complex(const float real, const float imaginary)
: real_(real), imaginary_(imaginary)
{}

__device__ Complex Complex::operator*(const Complex& rhs) const
{
    return Complex(real_ * rhs.real_ - imaginary_ * rhs.imaginary_, real_ * rhs.imaginary_ + imaginary_ * rhs.real_);
}

__device__ Complex Complex::operator+(const Complex& rhs) const
{
    return Complex(real_ + rhs.real_, imaginary_ + rhs.imaginary_);
}

__device__ float Complex::GetMagnitudeSquare(void) const
{
    return real_ * real_ + imaginary_ * imaginary_;
}

__device__ bool Complex::DoesJuliaExpansionConverge(const std::size_t length, const float threshold) const
{
    Complex z(this->real_, this->imaginary_);
    Complex constant(-0.8, 0.156);

    for (std::size_t it = 0; it < length; ++it)
    {
        z = z * z + constant;
        if (z.GetMagnitudeSquare() > threshold)
        {
            return false;
        }
    }

    return true;
}

__global__ void kernel(std::uint8_t* p_image)
{
    float real = static_cast<float>((int)blockIdx.x - (int)gridDim.x / 2) / (gridDim.x / 2);
    float imaginary = static_cast<float>((int)gridDim.y / 2 - (int)blockIdx.y) / (gridDim.y / 2);

    Complex z(real, imaginary);

    int offset = blockIdx.y * (4 * gridDim.x) + 4 * blockIdx.x;
    p_image[offset + 0] = z.DoesJuliaExpansionConverge() ? 255 : 0;
    p_image[offset + 1] = 0;
    p_image[offset + 2] = 0;
    p_image[offset + 3] = 255;
}