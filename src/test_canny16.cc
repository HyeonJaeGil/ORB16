#include "canny16.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;

int main(int argc, char **argv) {

  // load images from assets
  Mat img = imread("../assets/lena.png", IMREAD_GRAYSCALE);
  Mat img16 = imread("../assets/tir.png", IMREAD_UNCHANGED);

  // detect Canny edges
  cv::Mat edges;
  cv::Canny16(img, edges, 50, 100);

  // detect Canny edges on 16bit image
  cv::Mat edges16;
  cv::Canny16(img16, edges16, 100, 200);

  // detect Canny edges on 16bit image with explicit dx, dy
  cv::Mat dx, dy;
  cv::Sobel(img16, dx, CV_64F, 1, 0, 3);
  cv::Sobel(img16, dy, CV_64F, 0, 1, 3);
  cv::Mat edges16_dx_dy;
  cv::Canny16(dx, dy, edges16_dx_dy, 100, 200);

  // visualize
  cv::imshow("edges", edges);
  cv::imshow("edges16", edges16);
  cv::imshow("edges16_dx_dy", edges16_dx_dy);
  cv::waitKey(0);

  return 0;
}