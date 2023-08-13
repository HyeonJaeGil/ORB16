#include "ORB16.h"
#include "fast16.h"
#include <iostream>
#include <opencv2/features2d.hpp>
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

std::string int8UToBinaryString(uint8_t value) {
  std::string binaryString;
  for (int bit = 7; bit >= 0; bit--) {
    binaryString += ((value >> bit) & 1) ? "1" : "0";
  }
  return binaryString;
}

std::string int8UArrayToBinaryString(uint8_t *values, int size) {
  std::string binaryString;
  for (int i = 0; i < size; i++) {
    binaryString += int8UToBinaryString(values[i]);
  }
  return binaryString;
}

bool testDetection(const Mat &img, const Mat &img8) {

  // detect orb 16bit
  auto orb16 = ORB16::create(500, 1.2f, 8, 31, 0, 2, ORB16::HARRIS_SCORE, 31, 20);
  auto orb8 = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  std::vector<KeyPoint> keypoints_16;
  orb16->detect(img, keypoints_16);
  std::cout << "keypoints_16 size: " << keypoints_16.size() << std::endl;

  // detect orb 8bit
  std::vector<KeyPoint> keypoints_8;
  orb8->detect(img8, keypoints_8);
  std::cout << "keypoints_8 size: " << keypoints_8.size() << std::endl;

  // draw keypoints
  Mat img_keypoints_16;
  drawKeypoints(img8, keypoints_16, img_keypoints_16);
  imshow("orb 16 keypoints", img_keypoints_16);

  Mat img_keypoints_8;
  drawKeypoints(img8, keypoints_8, img_keypoints_8);
  imshow("orb 8 keypoints", img_keypoints_8);

  waitKey(0);
  return true;
}

bool testDescription(const Mat &img, const Mat &img8, std::vector<KeyPoint> keypoints) {
  auto orb16 = ORB16::create(500, 1.2f, 8, 31, 0, 2, ORB16::HARRIS_SCORE, 31, 20);
  auto orb8 = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

  if (keypoints.empty()) {
    orb16->detect(img, keypoints);
  }

  // select first 1 keypoints
  keypoints.resize(1);

  Mat descriptors_16;
  orb16->compute(img, keypoints, descriptors_16);
  auto binary16 = int8UArrayToBinaryString(descriptors_16.data, descriptors_16.cols);
  std::cout << "descriptor (orb16): \n" << binary16 << std::endl;

  Mat descriptors_8;
  orb8->compute(img8, keypoints, descriptors_8);
  auto binary8 = int8UArrayToBinaryString(descriptors_8.data, descriptors_8.cols);
  std::cout << "descriptor (orb8): \n" << binary8 << std::endl;

  return true;
}

int main(int argc, char **argv) {

  // load 16bit image from assets
  Mat img = imread("../assets/tir.png", IMREAD_UNCHANGED);
  Mat img8 = img.clone();

  // normalize img with min max value
  double min, max;
  cv::minMaxLoc(img, &min, &max);
  img.convertTo(img8, CV_8UC1, 255.0 / (max - min), -min * 255.0 / (max - min));

  // orb detection test
  bool detection_success = testDetection(img, img8);
  if (!detection_success) {
    std::cout << "detection failed" << std::endl;
    return -1;
  }

  // orb description test
  bool description_success = testDescription(img, img8, std::vector<KeyPoint>());
  if (!description_success) {
    std::cout << "description failed" << std::endl;
    return -1;
  }

  return 0;
}