#include "fast16.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
  cv::Mat img_half;
  cv::resize(img, img_half, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

  // test with fast16 detector
  auto fast16 = FastFeatureDetector16::create(30, true, FastFeatureDetector16::TYPE_9_16);
  std::vector<KeyPoint> keypoints_16;
  fast16->detect(img_half, keypoints_16);
  resizeKepoints(keypoints_16, 2.0); // half size compensation
  std::cout << "keypoints_16 size: " << keypoints_16.size() << std::endl;

  // test with fast12 detector
  auto fast12 = FastFeatureDetector16::create(30, true, FastFeatureDetector16::TYPE_7_12);
  std::vector<KeyPoint> keypoints_12;
  fast12->detect(img_half, keypoints_12);
  resizeKepoints(keypoints_12, 2.0); // half size compensation
  std::cout << "keypoints_12 size: " << keypoints_12.size() << std::endl;

  // test with fast8 detector
  auto fast8 = FastFeatureDetector16::create(30, true, FastFeatureDetector16::TYPE_5_8);
  std::vector<KeyPoint> keypoints_8;
  fast8->detect(img_half, keypoints_8);
  resizeKepoints(keypoints_8, 2.0); // half size compensation
  std::cout << "keypoints_8 size: " << keypoints_8.size() << std::endl;

  // draw keypoints
  Mat img_keypoints_16, img_keypoints_12, img_keypoints_8;
  drawKeypoints(img8, keypoints_16, img_keypoints_16);
  drawKeypoints(img8, keypoints_12, img_keypoints_12);
  drawKeypoints(img8, keypoints_8, img_keypoints_8);
  imshow("keypoints_16 (half)", img_keypoints_16);
  imshow("keypoints_12 (half)", img_keypoints_12);
  imshow("keypoints_8 (half)", img_keypoints_8);
  waitKey(0);

  return 0;
}