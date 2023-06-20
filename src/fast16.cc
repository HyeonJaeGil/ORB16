#include "fast16.h"
#include "buffer_area.hpp"
#include "fast16_score.h"
#include <iostream>

#define MAX16 65535

namespace cv {

template <int patternSize>
void FAST_t(InputArray _img, std::vector<KeyPoint> &keypoints, int threshold,
            bool nonmax_suppression) {
  Mat img = _img.getMat();
  const int K = patternSize / 2, N = patternSize + K + 1;
  int i, j, k, pixel[25];
  makeOffsets(pixel, (int)(img.step / 2),
              patternSize); // make stride the same as img.cols

  keypoints.clear();

  threshold = std::min(std::max(threshold, 0), MAX16);

  uchar threshold_tab[2 * (MAX16 + 1)];
  for (i = -MAX16; i <= MAX16; i++)
    threshold_tab[i + MAX16] =
        (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

  uchar *buf[3] = {0};
  int *cpbuf[3] = {0};
  utils::BufferArea area;
  for (unsigned idx = 0; idx < 3; ++idx) {
    area.allocate(buf[idx], img.cols);
    area.allocate(cpbuf[idx], img.cols + 1);
  }
  area.commit();

  for (unsigned idx = 0; idx < 3; ++idx) {
    memset(buf[idx], 0, img.cols);
  }

  for (i = 3; i < img.rows - 2;
       i++) // for each row (the first 3 rows are skipped)
  {
    const ushort *ptr =
        img.ptr<ushort>(i) + 3; // i-th row array, starting from the 4th column
    uchar *curr = buf[(i - 3) % 3];
    int *cornerpos =
        cpbuf[(i - 3) % 3] + 1; // cornerpos[-1] is used to store a value
    memset(curr, 0, img.cols);
    int ncorners = 0;

    if (i < img.rows - 3) {
      j = 3; // always start from 4th column of each row
      for (; j < img.cols - 3; j++, ptr++) {
        int v = ptr[0];
        const uchar *tab = &threshold_tab[0] - v + MAX16;
        for (int i = 0; i < 16; ++i) {
          // printf("pixel[%d]: %d, ptr[pixel[%d]]: %d\n", i, pixel[i], i,
          // ptr[pixel[i]]);
        }
        int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

        if (d == 0)
          continue;

        d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
        d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
        d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

        if (d == 0)
          continue;

        d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
        d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
        d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
        d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

        if (d & 1) {
          int vt = v - threshold, count = 0;

          for (k = 0; k < N; k++) {
            int x = ptr[pixel[k]];
            if (x < vt) {
              if (++count > K) {
                cornerpos[ncorners++] = j;
                if (nonmax_suppression)
                  curr[j] =
                      (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                break;
              }
            } else
              count = 0;
          }
        }

        if (d & 2) {
          int vt = v + threshold, count = 0;

          for (k = 0; k < N; k++) {
            int x = ptr[pixel[k]];
            if (x > vt) {
              if (++count > K) {
                cornerpos[ncorners++] = j;
                if (nonmax_suppression)
                  curr[j] =
                      (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                break;
              }
            } else
              count = 0;
          }
        }
      }
    }

    cornerpos[-1] = ncorners;

    if (i == 3)
      continue;

    const uchar *prev = buf[(i - 4 + 3) % 3];
    const uchar *pprev = buf[(i - 5 + 3) % 3];
    cornerpos =
        cpbuf[(i - 4 + 3) % 3] + 1; // cornerpos[-1] is used to store a value
    ncorners = cornerpos[-1];

    for (k = 0; k < ncorners; k++) {
      j = cornerpos[k];
      int score = prev[j];
      if (!nonmax_suppression ||
          (score > prev[j + 1] && score > prev[j - 1] && score > pprev[j - 1] &&
           score > pprev[j] && score > pprev[j + 1] && score > curr[j - 1] &&
           score > curr[j] && score > curr[j + 1])) {
        keypoints.push_back(
            KeyPoint((float)j, (float)(i - 1), 7.f, -1, (float)score));
      }
    }
  }
}

void FastFeatureDetector16::FAST(InputArray _img,
                                 std::vector<KeyPoint> &keypoints,
                                 int threshold, bool nonmax_suppression,
                                 FastFeatureDetector16::DetectorType type) {
  switch (type) {
  case FastFeatureDetector16::DetectorType::TYPE_5_8:
    FAST_t<8>(_img, keypoints, threshold, nonmax_suppression);
    break;
  case FastFeatureDetector16::DetectorType::TYPE_7_12:
    FAST_t<12>(_img, keypoints, threshold, nonmax_suppression);
    break;
  case FastFeatureDetector16::DetectorType::TYPE_9_16:
    FAST_t<16>(_img, keypoints, threshold, nonmax_suppression);
    break;
  }
}

void FastFeatureDetector16::FAST(InputArray _img,
                                 std::vector<KeyPoint> &keypoints,
                                 int threshold, bool nonmax_suppression) {
  FAST(_img, keypoints, threshold, nonmax_suppression,
       FastFeatureDetector16::DetectorType::TYPE_9_16);
}

void FastFeatureDetector16::detect(InputArray _image,
                                   std::vector<KeyPoint> &keypoints,
                                   InputArray _mask) {
  if (_image.empty()) {
    keypoints.clear();
    return;
  }
  Mat mask = _mask.getMat(), grayImage;
  UMat ugrayImage;
  _InputArray gray = _image;

  // if image type is not 16 bit unsigned, return error
  if (_image.type() % 8 != 2) {
    CV_Error(Error::StsBadArg, "image type must be 16 bit");
  }

  if (_image.type() != CV_16UC1) {
    _image.getMat().convertTo(grayImage, CV_16UC1);
    gray = grayImage;
  }

  FAST(gray, keypoints, threshold, nonmaxSuppression, type);
  KeyPointsFilter::runByPixelsMask(keypoints, mask);
}

void FastFeatureDetector16::set(int prop, double value) {
  if (prop == THRESHOLD)
    threshold = cvRound(value);
  else if (prop == NONMAX_SUPPRESSION)
    nonmaxSuppression = value != 0;
  else if (prop == FAST_N)
    type = static_cast<FastFeatureDetector16::DetectorType>(cvRound(value));
  else
    CV_Error(Error::StsBadArg, "");
}

double FastFeatureDetector16::get(int prop) const {
  if (prop == THRESHOLD)
    return threshold;
  if (prop == NONMAX_SUPPRESSION)
    return nonmaxSuppression;
  if (prop == FAST_N)
    return static_cast<int>(type);
  CV_Error(Error::StsBadArg, "");
  return 0;
}

} // namespace cv