/* This is FAST corner detector, contributed to OpenCV by the author, Edward
   Rosten. Below is the original copyright and the references */

/*
Copyright (c) 2006, 2008 Edward Rosten
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

    *Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

    *Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

    *Neither the name of the University of Cambridge nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
The references are:
 * Machine learning for high-speed corner detection,
   E. Rosten and T. Drummond, ECCV 2006
 * Faster and better: A machine learning approach to corner detection
   E. Rosten, R. Porter and T. Drummond, PAMI, 2009
*/

#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <algorithm>

namespace cv {

class FastFeatureDetector16 {
public:
  enum DetectorType { TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2 };
  enum { THRESHOLD = 10000, NONMAX_SUPPRESSION = 10001, FAST_N = 10002 };

  FastFeatureDetector16(int _threshold, bool _nonmaxSuppression,
                        FastFeatureDetector16::DetectorType _type)
      : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression),
        type(_type) {}

  static Ptr<FastFeatureDetector16>
  create(int threshold = 10, bool nonmaxSuppression = true,
         FastFeatureDetector16::DetectorType type =
             FastFeatureDetector16::TYPE_9_16) {
    return makePtr<FastFeatureDetector16>(threshold, nonmaxSuppression, type);
  }

  void FAST(InputArray _img, std::vector<KeyPoint> &keypoints, int threshold,
            bool nonmax_suppression, FastFeatureDetector16::DetectorType type);

  void FAST(InputArray _img, std::vector<KeyPoint> &keypoints, int threshold,
            bool nonmax_suppression);

  void detect(InputArray _image, std::vector<KeyPoint> &keypoints,
              InputArray _mask = noArray());

  void set(int prop, double value);

  double get(int prop) const;

  void setThreshold(int threshold_) { threshold = threshold_; }
  int getThreshold() const { return threshold; }

  void setNonmaxSuppression(bool f) { nonmaxSuppression = f; }
  bool getNonmaxSuppression() const { return nonmaxSuppression; }

  void setType(FastFeatureDetector16::DetectorType type_) { type = type_; }
  FastFeatureDetector16::DetectorType getType() const { return type; }
  String getDefaultName() const { return "FAST"; }

  int threshold;
  bool nonmaxSuppression;
  FastFeatureDetector16::DetectorType type;
};

} // namespace cv