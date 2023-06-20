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

  // detection with fast detector 16bit, half size
  auto fast16 =
      FastFeatureDetector16::create(40, true, FastFeatureDetector16::TYPE_9_16);
  std::vector<KeyPoint> keypoints_16_half;
  fast16->detect(img_half, keypoints_16_half);
  resizeKepoints(keypoints_16_half, 2.0); // half size compensation
  std::cout << "keypoints_16_half size: " << keypoints_16_half.size() << std::endl;

  // detection with fast detector 16bit
  std::vector<KeyPoint> keypoints_16;
  fast16->detect(img, keypoints_16);
  std::cout << "keypoints_16 size: " << keypoints_16.size() << std::endl;

  // detection with fast detector 8bit, half size
  auto fast8 = FastFeatureDetector::create(10, true);
  std::vector<KeyPoint> keypoints_8_half;
  fast8->detect(img8_half, keypoints_8_half);
  resizeKepoints(keypoints_8_half, 2.0); // half size compensation
  std::cout << "keypoints_8_half size: " << keypoints_8_half.size() << std::endl;

  // detection with fast detector 8bit
  std::vector<KeyPoint> keypoints_8;
  fast8->detect(img8, keypoints_8);
  std::cout << "keypoints_8 size: " << keypoints_8.size() << std::endl;

  // draw keypoints
  Mat img_keypoints_16_half;
  Mat img_keypoints_16;
  Mat img_keypoints_8_half;
  Mat img_keypoints_8;
  drawKeypoints(img8, keypoints_16_half, img_keypoints_16_half);
  drawKeypoints(img8, keypoints_16, img_keypoints_16);
  drawKeypoints(img8, keypoints_8_half, img_keypoints_8_half);
  drawKeypoints(img8, keypoints_8, img_keypoints_8);
  putText(img_keypoints_16_half, "fast 16bit (half) :"+std::to_string(keypoints_16_half.size()), Point(10, 30),
          FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
  putText(img_keypoints_16, "fast 16bit :"+std::to_string(keypoints_16.size()), Point(10, 30), FONT_HERSHEY_SIMPLEX,
          0.7, Scalar(255, 255, 255), 2);
  putText(img_keypoints_8_half, "fast 8bit (half) :" + std::to_string(keypoints_8_half.size()), Point(10, 30),
          FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
  putText(img_keypoints_8, "fast 8bit :" + std::to_string(keypoints_8.size()), Point(10, 30), FONT_HERSHEY_SIMPLEX,
          0.7, Scalar(255, 255, 255), 2);
  
  Mat concat = Mat::zeros(img_keypoints_16_half.rows * 2, img_keypoints_16_half.cols * 2, CV_8UC3);
  img_keypoints_16_half.copyTo(concat(Rect(0, 0, img_keypoints_16_half.cols, img_keypoints_16_half.rows)));
  img_keypoints_16.copyTo(concat(Rect(img_keypoints_16_half.cols, 0, img_keypoints_16.cols, img_keypoints_16.rows)));
  img_keypoints_8_half.copyTo(concat(Rect(0, img_keypoints_16_half.rows, img_keypoints_8_half.cols, img_keypoints_8_half.rows)));
  img_keypoints_8.copyTo(concat(Rect(img_keypoints_16_half.cols, img_keypoints_16_half.rows, img_keypoints_8.cols, img_keypoints_8.rows)));

  imshow("concat", concat);

  waitKey(0);

  return 0;
}