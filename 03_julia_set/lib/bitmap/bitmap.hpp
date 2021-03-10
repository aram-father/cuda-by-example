/**
 * @file bitmap.hpp
 * @author Wonseok Lee (aram_father@naver.com)
 * @brief Bitmap
 * @version 0.1
 * @date 2021-03-08
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef __BITMAP_HPP__
#define __BITMAP_HPP__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

class Bitmap
{
private:
  cv::Mat* image_;

public:
  Bitmap(std::size_t nrow, std::size_t ncol);
  ~Bitmap(void);

  const cv::Mat& image(void) const;
  cv::Mat& image(void);

  void ShowAndWait(const std::string& wnd = "test");
};

#endif