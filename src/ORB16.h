#pragma once
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace cv {

/** @brief Class implementing the ORB16 (*oriented BRIEF*) keypoint detector and
descriptor extractor

described in @cite RRKB11 . The algorithm uses FAST in pyramids to detect stable
keypoints, selects the strongest features using FAST or Harris response, finds
their orientation using first-order moments and computes the descriptors using
BRIEF (where the coordinates of random point pairs (or k-tuples) are rotated
according to the measured orientation).
 */
class ORB16 {
public:
  enum ScoreType { HARRIS_SCORE = 0, FAST_SCORE = 1 };
  static const int kBytes = 32;

  /** @brief The ORB16 constructor

  @param nfeatures The maximum number of features to retain.
  @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2
  means the classical pyramid, where each next level has 4x less pixels than the
  previous, but such a big scale factor will degrade feature matching scores
  dramatically. On the other hand, too close to 1 scale factor will mean that to
  cover certain scale range you will need more pyramid levels and so the speed
  will suffer.
  @param nlevels The number of pyramid levels. The smallest level will have
  linear size equal to input_image_linear_size/pow(scaleFactor, nlevels -
  firstLevel).
  @param edgeThreshold This is size of the border where the features are not
  detected. It should roughly match the patchSize parameter.
  @param firstLevel The level of pyramid to put source image to. Previous layers
  are filled with upscaled source image.
  @param WTA_K The number of points that produce each element of the oriented
  BRIEF descriptor. The default value 2 means the BRIEF where we take a random
  point pair and compare their brightnesses, so we get 0/1 response. Other
  possible values are 3 and 4. For example, 3 means that we take 3 random points
  (of course, those point coordinates are random, but they are generated from
  the pre-defined seed, so each element of BRIEF descriptor is computed
  deterministically from the pixel rectangle), find point of maximum brightness
  and output index of the winner (0, 1 or 2). Such output will occupy 2 bits,
  and therefore it will need a special variant of Hamming distance, denoted as
  NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to
  compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or
  3).
  @param scoreType The default HARRIS_SCORE means that Harris algorithm is used
  to rank features (the score is written to KeyPoint::score and is used to
  retain best nfeatures features); FAST_SCORE is alternative value of the
  parameter that produces slightly less stable keypoints, but it is a little
  faster to compute.
  @param patchSize size of the patch used by the oriented BRIEF descriptor. Of
  course, on smaller pyramid layers the perceived image area covered by a
  feature will be larger.
  @param fastThreshold the fast threshold
   */
  explicit ORB16(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
                 int _firstLevel, int _WTA_K, ORB16::ScoreType _scoreType, int _patchSize,
                 int _fastThreshold)
      : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        edgeThreshold(_edgeThreshold), firstLevel(_firstLevel), wta_k(_WTA_K),
        scoreType(_scoreType), patchSize(_patchSize), fastThreshold(_fastThreshold) {}

  static Ptr<ORB16> create(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8,
                           int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2,
                           ORB16::ScoreType scoreType = ORB16::HARRIS_SCORE, int patchSize = 31,
                           int fastThreshold = 20) {
    CV_Assert(firstLevel >= 0);
    return makePtr<ORB16>(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
                          scoreType, patchSize, fastThreshold);
  }

  void setMaxFeatures(int maxFeatures) { nfeatures = maxFeatures; }
  int getMaxFeatures() const { return nfeatures; }

  void setScaleFactor(double scaleFactor_) { scaleFactor = scaleFactor_; }
  double getScaleFactor() const { return scaleFactor; }

  void setNLevels(int nlevels_) { nlevels = nlevels_; }
  int getNLevels() const { return nlevels; }

  void setEdgeThreshold(int edgeThreshold_) { edgeThreshold = edgeThreshold_; }
  int getEdgeThreshold() const { return edgeThreshold; }

  void setFirstLevel(int firstLevel_) {
    CV_Assert(firstLevel_ >= 0);
    firstLevel = firstLevel_;
  }
  int getFirstLevel() const { return firstLevel; }

  void setWTA_K(int wta_k_) { wta_k = wta_k_; }
  int getWTA_K() const { return wta_k; }

  void setScoreType(ORB16::ScoreType scoreType_) { scoreType = scoreType_; }
  ORB16::ScoreType getScoreType() const { return scoreType; }

  void setPatchSize(int patchSize_) { patchSize = patchSize_; }
  int getPatchSize() const { return patchSize; }

  void setFastThreshold(int fastThreshold_) { fastThreshold = fastThreshold_; }
  int getFastThreshold() const { return fastThreshold; }
  String getDefaultName() const { return "ORB16"; }

  // returns the descriptor size in bytes
  int descriptorSize() const { return kBytes; }
  // returns the descriptor type
  int descriptorType() const { return CV_8U; }
  // returns the default norm type
  int defaultNorm() const {
    switch (wta_k) {
    case 2:
      return NORM_HAMMING;
    case 3:
    case 4:
      return NORM_HAMMING2;
    default:
      return -1;
    }
  }

  /** @brief Detects keypoints in an image (first variant) or image set (second
  variant).

  @param image Image.
  @param keypoints The detected keypoints. In the second variant of the method
  keypoints[i] is a set of keypoints detected in images[i] .
  @param mask Mask specifying where to look for keypoints (optional). It must be
  a 8-bit integer matrix with non-zero values in the region of interest.
   */
  virtual void detect(InputArray image, CV_OUT std::vector<KeyPoint> &keypoints,
                      InputArray mask = noArray());

  /** @overload
  @param images Image set.
  @param keypoints The detected keypoints. In the second variant of the method
  keypoints[i] is a set of keypoints detected in images[i] .
  @param masks Masks for each input image specifying where to look for keypoints
  (optional). masks[i] is a mask for images[i].
  */
  virtual void detect(InputArrayOfArrays images,
                      CV_OUT std::vector<std::vector<KeyPoint>> &keypoints,
                      InputArrayOfArrays masks = noArray());

  /** @brief Computes the descriptors for a set of keypoints detected in an
  image (first variant) or image set (second variant).

  @param image Image.
  @param keypoints Input collection of keypoints. Keypoints for which a
  descriptor cannot be computed are removed. Sometimes new keypoints can be
  added, for example: SIFT duplicates keypoint with several dominant
  orientations (for each orientation).
  @param descriptors Computed descriptors. In the second variant of the method
  descriptors[i] are descriptors computed for a keypoints[i]. Row j is the
  keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
   */
  virtual void compute(InputArray image, CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                       OutputArray descriptors);

  /** @overload

  @param images Image set.
  @param keypoints Input collection of keypoints. Keypoints for which a
  descriptor cannot be computed are removed. Sometimes new keypoints can be
  added, for example: SIFT duplicates keypoint with several dominant
  orientations (for each orientation).
  @param descriptors Computed descriptors. In the second variant of the method
  descriptors[i] are descriptors computed for a keypoints[i]. Row j is the
  keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
  */
  virtual void compute(InputArrayOfArrays images,
                       CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint>> &keypoints,
                       OutputArrayOfArrays descriptors);

  /** Detects keypoints and computes the descriptors */
  virtual void detectAndCompute(InputArray image, InputArray mask,
                                CV_OUT std::vector<KeyPoint> &keypoints, OutputArray descriptors,
                                bool useProvidedKeypoints = false);

protected:
  int nfeatures;
  double scaleFactor;
  int nlevels;
  int edgeThreshold;
  int firstLevel;
  int wta_k;
  ORB16::ScoreType scoreType;
  int patchSize;
  int fastThreshold;
};

} // namespace cv