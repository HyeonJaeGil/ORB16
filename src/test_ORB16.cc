#include "ORB16.h"
#include <iostream>

using namespace cv;

int main(int argc, char **argv) {

  // load image from assets
  Mat img = imread("../assets/lena.png", IMREAD_GRAYSCALE);
  cv::imshow("lena", img);
  cv::waitKey(0);

  auto orb = ORB16::create(500, 1.2f, 8, 31, 0, 2, ORB16::HARRIS_SCORE, 31, 20);

  // detect keypoints
  std::vector<KeyPoint> keypoints;
  orb->detect(img, keypoints);

  // draw keypoints
  Mat img_keypoints;
  drawKeypoints(img, keypoints, img_keypoints);
  imshow("Keypoints", img_keypoints);
  waitKey(0);

  // compute descriptors
  Mat descriptors;
  orb->compute(img, keypoints, descriptors);
  std::cout << "descriptor size: " << descriptors.size() << std::endl;

  return 0;
}