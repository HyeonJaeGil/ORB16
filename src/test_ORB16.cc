#include "ORB16.h"
#include "fast16.h"
#include <iostream>

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

  // resize to half
  cv::Mat img_half, img8_half;
  cv::resize(img, img_half, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
  cv::resize(img8, img8_half, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

  ////////////////////////////////////////////////////////////////////////////

  // test with fast detector 16bit, half size
  auto fast =
      FastFeatureDetector16::create(30, true, FastFeatureDetector16::TYPE_9_16);
  std::vector<KeyPoint> keypoints_16;
  fast->detect(img_half, keypoints_16);
  resizeKepoints(keypoints_16, 2.0); // half size compensation
  std::cout << "keypoints_16 size: " << keypoints_16.size() << std::endl;

  // draw keypoints
  Mat img_keypoints_16;
  drawKeypoints(img8, keypoints_16, img_keypoints_16);
  imshow("keypoints_16 (half)", img_keypoints_16);
  waitKey(0);

  ////////////////////////////////////////////////////////////////////////////

  // test with fast detector 8bit, half size
  auto orb = ORB16::create(500, 1.2f, 8, 31, 0, 2, ORB16::HARRIS_SCORE, 31, 7);
  std::vector<KeyPoint> keypoints_8_half;
  orb->detect(img8_half, keypoints_8_half);
  resizeKepoints(keypoints_8_half, 2.0); // half size compensation
  std::cout << "keypoints_8_half size: " << keypoints_8_half.size()
            << std::endl;

  // draw keypoints
  Mat img_keypoints_8_half;
  drawKeypoints(img8, keypoints_8_half, img_keypoints_8_half);
  imshow("keypoints_8 (half)", img_keypoints_8_half);
  waitKey(0);

  ////////////////////////////////////////////////////////////////////////////

  // test with fast detector 8bit (ORB default)
  std::vector<KeyPoint> keypoints_8;
  orb->detect(img8, keypoints_8);
  std::cout << "keypoints_8 size: " << keypoints_8.size() << std::endl;

  // draw keypoints
  Mat img_keypoints_8;
  drawKeypoints(img8, keypoints_8, img_keypoints_8);
  imshow("keypoints_8 (original)", img_keypoints_8);
  waitKey(0);

  // // compute ORB descriptors
  // Mat descriptors;
  // orb->compute(img, keypoints_16, descriptors);
  // std::cout << "descriptor size (16bit): " << descriptors.size() <<
  // std::endl; orb->compute(img8, keypoints_8, descriptors); std::cout <<
  // "descriptor size (8bit): " << descriptors.size() << std::endl;

  return 0;
}