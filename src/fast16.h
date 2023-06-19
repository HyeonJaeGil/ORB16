#pragma once
#include <opencv2/opencv.hpp>

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