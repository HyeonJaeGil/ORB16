#include "canny16.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>

using namespace cv;

int main(int argc, char **argv) {

  // load 8bit image from assets
  Mat img = imread("../assets/lena.png", IMREAD_UNCHANGED);

  // resize to half
  cv::Mat img_half;
  cv::resize(img, img_half, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

  // detect Canny edges
  cv::Mat edges;
  cv::Canny16(img_half, edges, 50, 100);

  // visualize
  cv::imshow("edges", edges);
  cv::waitKey(0);

  return 0;
}