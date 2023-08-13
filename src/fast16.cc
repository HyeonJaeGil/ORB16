#include "fast16.h"
#include "buffer_area.hpp"
#include <iostream>

#define VERIFY_CORNERS 0
#define MAX16 65535

namespace cv {

void makeOffsets(int pixel[25], int rowStride, int patternSize) {
  static const int offsets16[][2] = {{0, 3},  {1, 3},  {2, 2},  {3, 1},   {3, 0},   {3, -1},
                                     {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
                                     {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}};

  static const int offsets12[][2] = {{0, 2},  {1, 2},   {2, 1},   {2, 0},  {2, -1}, {1, -2},
                                     {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-1, 2}};

  static const int offsets8[][2] = {{0, 1},  {1, 1},   {1, 0},  {1, -1},
                                    {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};

  const int(*offsets)[2] = patternSize == 16 ?
                               offsets16 :
                               patternSize == 12 ? offsets12 : patternSize == 8 ? offsets8 : 0;

  CV_Assert(pixel && offsets);

  int k = 0;
  for (; k < patternSize; k++)
    pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
  for (; k < 25; k++)
    pixel[k] = pixel[k - patternSize];
}

#if VERIFY_CORNERS
static void testCorner(const ushort *ptr, const int pixel[], int K, int N, int threshold) {
  // check that with the computed "threshold" the pixel is still a corner
  // and that with the increased-by-1 "threshold" the pixel is not a corner
  // anymore
  for (int delta = 0; delta <= 1; delta++) {
    int v0 = std::min(ptr[0] + threshold + delta, 255);
    int v1 = std::max(ptr[0] - threshold - delta, 0);
    int c0 = 0, c1 = 0;

    for (int k = 0; k < N; k++) {
      int x = ptr[pixel[k]];
      if (x > v0) {
        if (++c0 > K)
          break;
        c1 = 0;
      } else if (x < v1) {
        if (++c1 > K)
          break;
        c0 = 0;
      } else {
        c0 = c1 = 0;
      }
    }
    CV_Assert((delta == 0 && std::max(c0, c1) > K) || (delta == 1 && std::max(c0, c1) <= K));
  }
}
#endif

template <int patternSize> int cornerScore(const ushort *ptr, const int pixel[], int threshold);

template <> int cornerScore<16>(const ushort *ptr, const int pixel[], int threshold) {
  const int K = 8, N = K * 3 + 1;
  int k, v = ptr[0];
  short d[N];
  for (k = 0; k < N; k++)
    d[k] = (short)(v - ptr[pixel[k]]);

  int a0 = threshold;
  for (k = 0; k < 16; k += 2) {
    int a = std::min((int)d[k + 1], (int)d[k + 2]);
    a = std::min(a, (int)d[k + 3]);
    if (a <= a0)
      continue;
    a = std::min(a, (int)d[k + 4]);
    a = std::min(a, (int)d[k + 5]);
    a = std::min(a, (int)d[k + 6]);
    a = std::min(a, (int)d[k + 7]);
    a = std::min(a, (int)d[k + 8]);
    a0 = std::max(a0, std::min(a, (int)d[k]));
    a0 = std::max(a0, std::min(a, (int)d[k + 9]));
  }

  int b0 = -a0;
  for (k = 0; k < 16; k += 2) {
    int b = std::max((int)d[k + 1], (int)d[k + 2]);
    b = std::max(b, (int)d[k + 3]);
    b = std::max(b, (int)d[k + 4]);
    b = std::max(b, (int)d[k + 5]);
    if (b >= b0)
      continue;
    b = std::max(b, (int)d[k + 6]);
    b = std::max(b, (int)d[k + 7]);
    b = std::max(b, (int)d[k + 8]);

    b0 = std::min(b0, std::max(b, (int)d[k]));
    b0 = std::min(b0, std::max(b, (int)d[k + 9]));
  }

  threshold = -b0 - 1;

#if VERIFY_CORNERS
  testCorner(ptr, pixel, K, N, threshold);
#endif
  return threshold;
}

template <> int cornerScore<12>(const ushort *ptr, const int pixel[], int threshold) {
  const int K = 6, N = K * 3 + 1;
  int k, v = ptr[0];
  short d[N + 4];
  for (k = 0; k < N; k++)
    d[k] = (short)(v - ptr[pixel[k]]);

  int a0 = threshold;
  for (k = 0; k < 12; k += 2) {
    int a = std::min((int)d[k + 1], (int)d[k + 2]);
    if (a <= a0)
      continue;
    a = std::min(a, (int)d[k + 3]);
    a = std::min(a, (int)d[k + 4]);
    a = std::min(a, (int)d[k + 5]);
    a = std::min(a, (int)d[k + 6]);
    a0 = std::max(a0, std::min(a, (int)d[k]));
    a0 = std::max(a0, std::min(a, (int)d[k + 7]));
  }

  int b0 = -a0;
  for (k = 0; k < 12; k += 2) {
    int b = std::max((int)d[k + 1], (int)d[k + 2]);
    b = std::max(b, (int)d[k + 3]);
    b = std::max(b, (int)d[k + 4]);
    if (b >= b0)
      continue;
    b = std::max(b, (int)d[k + 5]);
    b = std::max(b, (int)d[k + 6]);

    b0 = std::min(b0, std::max(b, (int)d[k]));
    b0 = std::min(b0, std::max(b, (int)d[k + 7]));
  }

  threshold = -b0 - 1;
#if VERIFY_CORNERS
  testCorner(ptr, pixel, K, N, threshold);
#endif
  return threshold;
}

template <> int cornerScore<8>(const ushort *ptr, const int pixel[], int threshold) {
  const int K = 4, N = K * 3 + 1;
  int k, v = ptr[0];
  short d[N];
  for (k = 0; k < N; k++)
    d[k] = (short)(v - ptr[pixel[k]]);

  int a0 = threshold;
  for (k = 0; k < 8; k += 2) {
    int a = std::min((int)d[k + 1], (int)d[k + 2]);
    if (a <= a0)
      continue;
    a = std::min(a, (int)d[k + 3]);
    a = std::min(a, (int)d[k + 4]);
    a0 = std::max(a0, std::min(a, (int)d[k]));
    a0 = std::max(a0, std::min(a, (int)d[k + 5]));
  }

  int b0 = -a0;
  for (k = 0; k < 8; k += 2) {
    int b = std::max((int)d[k + 1], (int)d[k + 2]);
    b = std::max(b, (int)d[k + 3]);
    if (b >= b0)
      continue;
    b = std::max(b, (int)d[k + 4]);

    b0 = std::min(b0, std::max(b, (int)d[k]));
    b0 = std::min(b0, std::max(b, (int)d[k + 5]));
  }

  threshold = -b0 - 1;

#if VERIFY_CORNERS
  testCorner(ptr, pixel, K, N, threshold);
#endif
  return threshold;
}

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
    threshold_tab[i + MAX16] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

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

  for (i = 3; i < img.rows - 2; i++) // for each row (the first 3 rows are skipped)
  {
    const ushort *ptr = img.ptr<ushort>(i) + 3; // i-th row array, starting from the 4th column
    uchar *curr = buf[(i - 3) % 3];
    int *cornerpos = cpbuf[(i - 3) % 3] + 1; // cornerpos[-1] is used to store a value
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
                  curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
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
                  curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
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
    cornerpos = cpbuf[(i - 4 + 3) % 3] + 1; // cornerpos[-1] is used to store a value
    ncorners = cornerpos[-1];

    for (k = 0; k < ncorners; k++) {
      j = cornerpos[k];
      int score = prev[j];
      if (!nonmax_suppression ||
          (score > prev[j + 1] && score > prev[j - 1] && score > pprev[j - 1] && score > pprev[j] &&
           score > pprev[j + 1] && score > curr[j - 1] && score > curr[j] && score > curr[j + 1])) {
        keypoints.push_back(KeyPoint((float)j, (float)(i - 1), 7.f, -1, (float)score));
      }
    }
  }
}

void FastFeatureDetector16::FAST(InputArray _img, std::vector<KeyPoint> &keypoints, int threshold,
                                 bool nonmax_suppression,
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

void FastFeatureDetector16::FAST(InputArray _img, std::vector<KeyPoint> &keypoints, int threshold,
                                 bool nonmax_suppression) {
  FAST(_img, keypoints, threshold, nonmax_suppression,
       FastFeatureDetector16::DetectorType::TYPE_9_16);
}

void FastFeatureDetector16::detect(InputArray _image, std::vector<KeyPoint> &keypoints,
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