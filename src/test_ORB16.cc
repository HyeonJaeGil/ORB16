#include "ORB16.h"
#include "fast16.h"
#include <iostream>

using namespace cv;

int main(int argc, char **argv) {

  // // load image from assets
  // Mat img = imread("../assets/lena.png", IMREAD_GRAYSCALE);
  // cv::imshow("lena", img);
  // cv::waitKey(0);

  // load 16bit image from assets
  Mat img = imread("../assets/tir.png", IMREAD_UNCHANGED);
  // cv::imshow("tir", img);
  // cv::waitKey(0);
  Mat img8 = img.clone();
  // normalize img with min max value
  double min, max;
  cv::minMaxLoc(img, &min, &max);
  img.convertTo(img8, CV_8UC1, 255.0 / (max - min), -min * 255.0 / (max - min));


  // test with fast detector
  auto fast = FastFeatureDetector16::create(200, true, FastFeatureDetector16::TYPE_9_16);
  std::vector<KeyPoint> keypoints_fast;
  fast->detect(img, keypoints_fast);
  std::cout << "keypoints_fast size: " << keypoints_fast.size() << std::endl;

  // draw keypoints
  Mat img_keypoints_fast;
  drawKeypoints(img8, keypoints_fast, img_keypoints_fast);
  imshow("Keypoints_fast", img_keypoints_fast);
  waitKey(0);


  auto orb = ORB16::create(500, 1.2f, 8, 31, 0, 2, ORB16::HARRIS_SCORE, 31, 20);

  // detect keypoints
  std::vector<KeyPoint> keypoints;
  orb->detect(img8, keypoints);

  // draw keypoints
  Mat img_keypoints;
  drawKeypoints(img8, keypoints, img_keypoints);
  imshow("Keypoints", img_keypoints);
  waitKey(0);

  // compute descriptors
  Mat descriptors;
  orb->compute(img, keypoints, descriptors);
  std::cout << "descriptor size: " << descriptors.size() << std::endl;

  return 0;
}