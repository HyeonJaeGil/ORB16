#include "canny16.h"
#include <deque>

namespace cv {

/** Canny edge detector flags */
enum { CV_CANNY_L2_GRADIENT = (1 << 31) };

#define CANNY_PUSH(map, stack) *map = 2, stack.push_back(map)

#define CANNY_CHECK(m, high, map, stack)                                                           \
  if (m > high)                                                                                    \
    CANNY_PUSH(map, stack);                                                                        \
  else                                                                                             \
    *map = 0

class parallelCanny16 : public ParallelLoopBody {
public:
  parallelCanny16(const Mat &_src, Mat &_map, std::deque<uchar *> &borderPeaksParallel, int _low,
                  int _high, int _aperture_size, bool _L2gradient)
      : src(_src), src2(_src), map(_map), _borderPeaksParallel(borderPeaksParallel), low(_low),
        high(_high), aperture_size(_aperture_size), L2gradient(_L2gradient) {
    _map.create(src.rows + 2, src.cols + 2, CV_8UC1);
    map = _map;
    map.row(0).setTo(1);
    map.row(src.rows + 1).setTo(1);
    mapstep = map.cols;
    needGradient = true;
    cn = src.channels();
  }

  parallelCanny16(const Mat &_dx, const Mat &_dy, Mat &_map,
                  std::deque<uchar *> &borderPeaksParallel, int _low, int _high, bool _L2gradient)
      : src(_dx), src2(_dy), map(_map), _borderPeaksParallel(borderPeaksParallel), low(_low),
        high(_high), aperture_size(0), L2gradient(_L2gradient) {
    _map.create(src.rows + 2, src.cols + 2, CV_8UC1);
    map = _map;
    map.row(0).setTo(1);
    map.row(src.rows + 1).setTo(1);
    mapstep = map.cols;
    needGradient = false;
    cn = src.channels();
  }

  ~parallelCanny16() {}

  parallelCanny16 &operator=(const parallelCanny16 &) { return *this; }

  void operator()(const Range &boundaries) const CV_OVERRIDE {
    CV_DbgAssert(cn > 0);

    Mat dx, dy;
    AutoBuffer<double> dxMax(0), dyMax(0);
    std::deque<uchar *> stack, borderPeaksLocal;
    const int rowStart = max(0, boundaries.start - 1), rowEnd = min(src.rows, boundaries.end + 1);
    int *_mag_p, *_mag_a, *_mag_n;
    double *_dx, *_dy, *_dx_a = NULL, *_dy_a = NULL, *_dx_n = NULL, *_dy_n = NULL;
    uchar *_pmap;
    double scale = 1.0;

    // CV_TRACE_REGION("gradient")
    if (needGradient) {
      if (aperture_size == 7) {
        scale = 1 / 16.0;
      }
      Sobel(src.rowRange(rowStart, rowEnd), dx, CV_64F, 1, 0, aperture_size, scale, 0,
            BORDER_REPLICATE);
      Sobel(src.rowRange(rowStart, rowEnd), dy, CV_64F, 0, 1, aperture_size, scale, 0,
            BORDER_REPLICATE);
    } else {
      dx = src.rowRange(rowStart, rowEnd);
      dy = src2.rowRange(rowStart, rowEnd);
    }

    // CV_TRACE_REGION_NEXT("magnitude");
    if (cn > 1) {
      dxMax.allocate(2 * dx.cols);
      dyMax.allocate(2 * dy.cols);
      _dx_a = dxMax.data();
      _dx_n = _dx_a + dx.cols;
      _dy_a = dyMax.data();
      _dy_n = _dy_a + dy.cols;
    }

    // _mag_p: previous row, _mag_a: actual row, _mag_n: next row

    AutoBuffer<int> buffer(3 * (mapstep * cn));
    _mag_p = buffer.data() + 1;
    _mag_a = _mag_p + mapstep * cn;
    _mag_n = _mag_a + mapstep * cn;

    // For the first time when just 2 rows are filled and for left and right borders
    if (rowStart == boundaries.start)
      memset(_mag_n - 1, 0, mapstep * sizeof(int));
    else
      _mag_n[src.cols] = _mag_n[-1] = 0;

    _mag_a[src.cols] = _mag_a[-1] = _mag_p[src.cols] = _mag_p[-1] = 0;

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = rowStart; i <= boundaries.end; ++i) {
      // Scroll the ring buffer
      std::swap(_mag_n, _mag_a);
      std::swap(_mag_n, _mag_p);

      if (i < rowEnd) {
        // Next row calculation
        _dx = dx.ptr<double>(i - rowStart);
        _dy = dy.ptr<double>(i - rowStart);

        if (L2gradient) {
          int j = 0, width = src.cols * cn;

          for (; j < width; ++j)
            _mag_n[j] = int(_dx[j]) * _dx[j] + int(_dy[j]) * _dy[j];
        } else {
          int j = 0, width = src.cols * cn;
          for (; j < width; ++j)
            _mag_n[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
        }

        if (cn > 1) {
          std::swap(_dx_n, _dx_a);
          std::swap(_dy_n, _dy_a);

          for (int j = 0, jn = 0; j < src.cols; ++j, jn += cn) {
            int maxIdx = jn;
            for (int k = 1; k < cn; ++k)
              if (_mag_n[jn + k] > _mag_n[maxIdx])
                maxIdx = jn + k;

            _mag_n[j] = _mag_n[maxIdx];
            _dx_n[j] = _dx[maxIdx];
            _dy_n[j] = _dy[maxIdx];
          }

          _mag_n[src.cols] = 0;
        }

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i <= boundaries.start)
          continue;
      } else {
        memset(_mag_n - 1, 0, mapstep * sizeof(int));

        if (cn > 1) {
          std::swap(_dx_n, _dx_a);
          std::swap(_dy_n, _dy_a);
        }
      }

      // From here actual src row is (i - 1)
      // Set left and right border to 1

      _pmap = map.ptr<uchar>(i) + 1;

      _pmap[src.cols] = _pmap[-1] = 1;

      if (cn == 1) {
        _dx = dx.ptr<double>(i - rowStart - 1);
        _dy = dy.ptr<double>(i - rowStart - 1);
      } else {
        _dx = _dx_a;
        _dy = _dy_a;
      }

      const int TG22 = 13573;
      int j = 0;

      for (; j < src.cols; j++) {
        int m = _mag_a[j];

        if (m > low) {
          double xs = _dx[j];
          double ys = _dy[j];
          int x = (int)std::abs(xs);
          int y = (int)std::abs(ys) << 15;

          int tg22x = x * TG22;

          if (y < tg22x) {
            if (m > _mag_a[j - 1] && m >= _mag_a[j + 1]) {
              CANNY_CHECK(m, high, (_pmap + j), stack);
              continue;
            }
          } else {
            int tg67x = tg22x + (x << 16);
            if (y > tg67x) {
              if (m > _mag_p[j] && m >= _mag_n[j]) {
                CANNY_CHECK(m, high, (_pmap + j), stack);
                continue;
              }
            } else {
              int s = ((xs < 0) != (ys < 0)) ? -1 : 1;
              if (m > _mag_p[j - s] && m > _mag_n[j + s]) {
                CANNY_CHECK(m, high, (_pmap + j), stack);
                continue;
              }
            }
          }
        }
        _pmap[j] = 1;
      }
    }

    // Not for first row of first slice or last row of last slice
    uchar *pmapLower = (rowStart == 0) ? map.data : (map.data + (boundaries.start + 2) * mapstep);
    uint pmapDiff = (uint)(
        ((rowEnd == src.rows) ? map.datalimit : (map.data + boundaries.end * mapstep)) - pmapLower);

    // now track the edges (hysteresis thresholding)
    // CV_TRACE_REGION_NEXT("hysteresis");
    while (!stack.empty()) {
      uchar *m = stack.back();
      stack.pop_back();

      // Stops thresholding from expanding to other slices by sending pixels in the borders of each
      // slice in a queue to be serially processed later.
      if ((unsigned)(m - pmapLower) < pmapDiff) {
        if (!m[-mapstep - 1])
          CANNY_PUSH((m - mapstep - 1), stack);
        if (!m[-mapstep])
          CANNY_PUSH((m - mapstep), stack);
        if (!m[-mapstep + 1])
          CANNY_PUSH((m - mapstep + 1), stack);
        if (!m[-1])
          CANNY_PUSH((m - 1), stack);
        if (!m[1])
          CANNY_PUSH((m + 1), stack);
        if (!m[mapstep - 1])
          CANNY_PUSH((m + mapstep - 1), stack);
        if (!m[mapstep])
          CANNY_PUSH((m + mapstep), stack);
        if (!m[mapstep + 1])
          CANNY_PUSH((m + mapstep + 1), stack);
      } else {
        borderPeaksLocal.push_back(m);
        ptrdiff_t mapstep2 = m < pmapLower ? mapstep : -mapstep;

        if (!m[-1])
          CANNY_PUSH((m - 1), stack);
        if (!m[1])
          CANNY_PUSH((m + 1), stack);
        if (!m[mapstep2 - 1])
          CANNY_PUSH((m + mapstep2 - 1), stack);
        if (!m[mapstep2])
          CANNY_PUSH((m + mapstep2), stack);
        if (!m[mapstep2 + 1])
          CANNY_PUSH((m + mapstep2 + 1), stack);
      }
    }

    if (!borderPeaksLocal.empty()) {
      AutoLock lock(mutex);
      _borderPeaksParallel.insert(_borderPeaksParallel.end(), borderPeaksLocal.begin(),
                                  borderPeaksLocal.end());
    }
  }

private:
  const Mat &src, &src2;
  Mat &map;
  std::deque<uchar *> &_borderPeaksParallel;
  int low, high, aperture_size;
  bool L2gradient, needGradient;
  ptrdiff_t mapstep;
  int cn;
  mutable Mutex mutex;
};

class finalPass16 : public ParallelLoopBody {

public:
  finalPass16(const Mat &_map, Mat &_dst) : map(_map), dst(_dst) { dst = _dst; }

  ~finalPass16() {}

  void operator()(const Range &boundaries) const CV_OVERRIDE {
    // the final pass, form the final image
    for (int i = boundaries.start; i < boundaries.end; i++) {
      int j = 0;
      uchar *pdst = dst.ptr<uchar>(i);
      const uchar *pmap = map.ptr<uchar>(i + 1);
      pmap += 1;

      for (; j < dst.cols; j++) {
        pdst[j] = (uchar) - (pmap[j] >> 1);
      }
    }
  }

private:
  const Mat &map;
  Mat &dst;

  finalPass16(const finalPass16 &);            // = delete
  finalPass16 &operator=(const finalPass16 &); // = delete
};

void Canny16(InputArray _src, OutputArray _dst, double low_thresh, double high_thresh,
             int aperture_size, bool L2gradient) {
  //   CV_Assert(_src.depth() == CV_8U);

  const Size size = _src.size();

  //   // we don't support inplace parameters in case with RGB/BGR src
  //   CV_Assert((_dst.getObj() != _src.getObj() || _src.type() == CV_8UC1) &&
  //             "Inplace parameters are not supported");

  _dst.create(size, CV_8U);

  if (!L2gradient && (aperture_size & CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT) {
    // backward compatibility
    aperture_size &= ~CV_CANNY_L2_GRADIENT;
    L2gradient = true;
  }

  if ((aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7)))
#if CV_MAJOR_VERSION >= 4
    CV_Error(Error::StsBadFlag, "Aperture size should be odd between 3 and 7");
#else
    CV_Error(CV_StsBadFlag, "Aperture size should be odd between 3 and 7");
#endif
  if (aperture_size == 7) {
    low_thresh = low_thresh / 16.0;
    high_thresh = high_thresh / 16.0;
  }

  if (low_thresh > high_thresh)
    std::swap(low_thresh, high_thresh);

  Mat src0 = _src.getMat(), dst = _dst.getMat();
  Mat src(src0.size(), src0.type(), src0.data, src0.step);

  if (L2gradient) {
    low_thresh = std::min(2147483647.0, low_thresh);
    high_thresh = std::min(2147483647.0, high_thresh);

    if (low_thresh > 0)
      low_thresh *= low_thresh;
    if (high_thresh > 0)
      high_thresh *= high_thresh;
  }
  int low = cvFloor(low_thresh);
  int high = cvFloor(high_thresh);

  // If Scharr filter: aperture size is 3, ksize2 is 1
  int ksize2 = aperture_size < 0 ? 1 : aperture_size / 2;
  // Minimum number of threads should be 1, maximum should not exceed number of CPU's, because of
  // overhead
  int numOfThreads = std::max(1, std::min(getNumThreads(), getNumberOfCPUs()));
  // Make a fallback for pictures with too few rows.
  int grainSize = src.rows / numOfThreads;
  int minGrainSize = 2 * (ksize2 + 1);
  if (grainSize < minGrainSize)
    numOfThreads = std::max(1, src.rows / minGrainSize);

  Mat map;
  std::deque<uchar *> stack;

  parallel_for_(Range(0, src.rows),
                parallelCanny16(src, map, stack, low, high, aperture_size, L2gradient),
                numOfThreads);

  // now track the edges (hysteresis thresholding)
  ptrdiff_t mapstep = map.cols;

  while (!stack.empty()) {
    uchar *m = stack.back();
    stack.pop_back();

    if (!m[-mapstep - 1])
      CANNY_PUSH((m - mapstep - 1), stack);
    if (!m[-mapstep])
      CANNY_PUSH((m - mapstep), stack);
    if (!m[-mapstep + 1])
      CANNY_PUSH((m - mapstep + 1), stack);
    if (!m[-1])
      CANNY_PUSH((m - 1), stack);
    if (!m[1])
      CANNY_PUSH((m + 1), stack);
    if (!m[mapstep - 1])
      CANNY_PUSH((m + mapstep - 1), stack);
    if (!m[mapstep])
      CANNY_PUSH((m + mapstep), stack);
    if (!m[mapstep + 1])
      CANNY_PUSH((m + mapstep + 1), stack);
  }

  parallel_for_(Range(0, src.rows), finalPass16(map, dst), src.total() / (double)(1 << 16));
}

void Canny16(InputArray _dx, InputArray _dy, OutputArray _dst, double low_thresh,
             double high_thresh, bool L2gradient) {
  CV_Assert(_dx.dims() == 2);
  CV_Assert(_dx.type() == CV_64FC1 || _dx.type() == CV_64FC3);
  CV_Assert(_dy.type() == _dx.type());
  CV_Assert(_dx.sameSize(_dy));

  if (low_thresh > high_thresh)
    std::swap(low_thresh, high_thresh);

  const Size size = _dx.size();

  _dst.create(size, CV_8U);
  Mat dst = _dst.getMat();

  Mat dx = _dx.getMat();
  Mat dy = _dy.getMat();

  if (L2gradient) {
    low_thresh = std::min(2147483647.0, low_thresh);
    high_thresh = std::min(2147483647.0, high_thresh);

    if (low_thresh > 0)
      low_thresh *= low_thresh;
    if (high_thresh > 0)
      high_thresh *= high_thresh;
  }

  int low = cvFloor(low_thresh);
  int high = cvFloor(high_thresh);

  std::deque<uchar *> stack;
  Mat map;

  // Minimum number of threads should be 1, maximum should not exceed number of CPU's, because of
  // overhead
  int numOfThreads = std::max(1, std::min(getNumThreads(), getNumberOfCPUs()));
  if (dx.rows / numOfThreads < 3)
    numOfThreads = std::max(1, dx.rows / 3);

  parallel_for_(Range(0, dx.rows), parallelCanny16(dx, dy, map, stack, low, high, L2gradient),
                numOfThreads);

  // CV_TRACE_REGION("global_hysteresis")
  // now track the edges (hysteresis thresholding)
  ptrdiff_t mapstep = map.cols;

  while (!stack.empty()) {
    uchar *m = stack.back();
    stack.pop_back();

    if (!m[-mapstep - 1])
      CANNY_PUSH((m - mapstep - 1), stack);
    if (!m[-mapstep])
      CANNY_PUSH((m - mapstep), stack);
    if (!m[-mapstep + 1])
      CANNY_PUSH((m - mapstep + 1), stack);
    if (!m[-1])
      CANNY_PUSH((m - 1), stack);
    if (!m[1])
      CANNY_PUSH((m + 1), stack);
    if (!m[mapstep - 1])
      CANNY_PUSH((m + mapstep - 1), stack);
    if (!m[mapstep])
      CANNY_PUSH((m + mapstep), stack);
    if (!m[mapstep + 1])
      CANNY_PUSH((m + mapstep + 1), stack);
  }

  // CV_TRACE_REGION_NEXT("finalPass16");
  parallel_for_(Range(0, dx.rows), finalPass16(map, dst), dx.total() / (double)(1 << 16));
}

} // namespace cv