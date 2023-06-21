#include "ORB16.h"
#include "fast16.h"
#include <iostream>
#include <opencv2/features2d.hpp>

using namespace cv;

void resizeKepoints(std::vector<KeyPoint> &keypoints, float scale) {
  for (auto &kp : keypoints) {
    kp.pt.x *= scale;
    kp.pt.y *= scale;
  }
}

int main(int argc, char **argv) {

  // load 16bit image from assets
  Mat img = imread("../assets/tir.png", IMREAD_UNCHANGED);
  Mat img8 = img.clone();

  // normalize img with min max value
  double min, max;
  cv::minMaxLoc(img, &min, &max);
  img.convertTo(img8, CV_8UC1, 255.0 / (max - min), -min * 255.0 / (max - min));

  // detect orb 16bit
  auto orb16 = ORB16::create(500, 1.2f, 8, 31, 0, 2, ORB16::HARRIS_SCORE, 31, 20);
  std::vector<KeyPoint> keypoints_16;
  orb16->detect(img, keypoints_16);
  std::cout << "keypoints_16 size: " << keypoints_16.size() << std::endl;
  Mat descriptors_16;
  orb16->compute(img, keypoints_16, descriptors_16); 
  std::cout << "descriptor size orb: " << descriptors_16.size() << std::endl;

  // detect orb 8bit
  auto orb8 = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  std::vector<KeyPoint> keypoints_8;
  orb8->detect(img8, keypoints_8);
  std::cout << "keypoints_8 size: " << keypoints_8.size() << std::endl;
  Mat descriptors_8;
  orb8->compute(img8, keypoints_8, descriptors_8);
  std::cout << "descriptor size orb: " << descriptors_8.size() << std::endl;


  // draw keypoints
  Mat img_keypoints_16;
  drawKeypoints(img8, keypoints_16, img_keypoints_16);
  imshow("orb 16 keypoints", img_keypoints_16);

  Mat img_keypoints_8;
  drawKeypoints(img8, keypoints_8, img_keypoints_8);
  imshow("orb 8 keypoints", img_keypoints_8);

  waitKey(0);


  return 0;
}