/**
 * @file bitmap.cc
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Bitmap
 * @version 0.1
 * @date 2021-03-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <opencv2/opencv.hpp>

#include "bitmap.hpp"

Bitmap::Bitmap(std::size_t nrow, std::size_t ncol)
{
  image_ = new cv::Mat(nrow, ncol, CV_8UC4);
}

Bitmap::~Bitmap(void)
{
  delete image_;
}

const cv::Mat& Bitmap::image(void) const
{
  return *image_;
}

cv::Mat& Bitmap::image(void)
{
  return const_cast<cv::Mat&>(const_cast<const Bitmap*>(this)->image());
}

void Bitmap::ShowAndWait(const std::string& wnd)
{
  cv::imshow(wnd, this->image());
  cv::waitKey(0);
}