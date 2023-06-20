#include "fast16.h"
#include "fast16_score.h"
#include <iostream>
#include "buffer_area.hpp"

#define MAX16 65535

namespace cv {

template<int patternSize>
void FAST_t(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    Mat img = _img.getMat();
    const int K = patternSize/2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)(img.step/2), patternSize); // make stride the same as img.cols

#if CV_SIMD128
    const int quarterPatternSize = patternSize/4;
    v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char)threshold), K16 = v_setall_u8((char)K);
#if CV_TRY_AVX2
    Ptr<opt_AVX2::FAST_t_patternSize16_AVX2> fast_t_impl_avx2;
    if(CV_CPU_HAS_SUPPORT_AVX2)
        fast_t_impl_avx2 = opt_AVX2::FAST_t_patternSize16_AVX2::getImpl(img.cols, threshold, nonmax_suppression, pixel);
#endif

#endif

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), MAX16);

    uchar threshold_tab[2*(MAX16+1)];
    for( i = -MAX16; i <= MAX16; i++ )
    {
        threshold_tab[i+MAX16] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);
        // printf("%d ", threshold_tab[i+MAX16]);
    }
    // printf("\n");
    

    uchar* buf[3] = { 0 };
    int* cpbuf[3] = { 0 };
    utils::BufferArea area;
    for (unsigned idx = 0; idx < 3; ++idx)
    {
        area.allocate(buf[idx], img.cols);
        area.allocate(cpbuf[idx], img.cols + 1);
    }
    area.commit();

    for (unsigned idx = 0; idx < 3; ++idx)
    {
        memset(buf[idx], 0, img.cols);
    }

    for(i = 3; i < img.rows-2; i++) // for each row (the first 3 rows are skipped)
    {
        const ushort* ptr = img.ptr<ushort>(i) + 3; // i-th row array, starting from the 4th column
        // // print each element of ptr
        // for (int i = 0; i < img.cols; i++)
        // {
        //     std::cout << (int)ptr[i] << " ";
        // }
        // std::cout << std::endl;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3] + 1; // cornerpos[-1] is used to store a value
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
        {
            j = 3; // always start from 4th column of each row
#if CV_SIMD128
            {
                if( patternSize == 16 )
                {
#if CV_TRY_AVX2
                    if (fast_t_impl_avx2)
                        fast_t_impl_avx2->process(j, ptr, curr, cornerpos, ncorners);
#endif
                    //vz if (j <= (img.cols - 27)) //it doesn't make sense using vectors for less than 8 elements
                    {
                        for (; j < img.cols - 16 - 3; j += 16, ptr += 16)
                        {
                            v_uint8x16 v = v_load(ptr);
                            v_int8x16 v0 = v_reinterpret_as_s8((v + t) ^ delta);
                            v_int8x16 v1 = v_reinterpret_as_s8((v - t) ^ delta);

                            v_int8x16 x0 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[0]), delta));
                            v_int8x16 x1 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[quarterPatternSize]), delta));
                            v_int8x16 x2 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[2*quarterPatternSize]), delta));
                            v_int8x16 x3 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[3*quarterPatternSize]), delta));

                            v_int8x16 m0, m1;
                            m0 = (v0 < x0) & (v0 < x1);
                            m1 = (x0 < v1) & (x1 < v1);
                            m0 = m0 | ((v0 < x1) & (v0 < x2));
                            m1 = m1 | ((x1 < v1) & (x2 < v1));
                            m0 = m0 | ((v0 < x2) & (v0 < x3));
                            m1 = m1 | ((x2 < v1) & (x3 < v1));
                            m0 = m0 | ((v0 < x3) & (v0 < x0));
                            m1 = m1 | ((x3 < v1) & (x0 < v1));
                            m0 = m0 | m1;

                            if( !v_check_any(m0) )
                                continue;
                            if( !v_check_any(v_combine_low(m0, m0)) )
                            {
                                j -= 8;
                                ptr -= 8;
                                continue;
                            }

                            v_int8x16 c0 = v_setzero_s8();
                            v_int8x16 c1 = v_setzero_s8();
                            v_uint8x16 max0 = v_setzero_u8();
                            v_uint8x16 max1 = v_setzero_u8();
                            for( k = 0; k < N; k++ )
                            {
                                v_int8x16 x = v_reinterpret_as_s8(v_load((ptr + pixel[k])) ^ delta);
                                m0 = v0 < x;
                                m1 = x < v1;

                                c0 = v_sub_wrap(c0, m0) & m0;
                                c1 = v_sub_wrap(c1, m1) & m1;

                                max0 = v_max(max0, v_reinterpret_as_u8(c0));
                                max1 = v_max(max1, v_reinterpret_as_u8(c1));
                            }

                            max0 = K16 < v_max(max0, max1);
                            unsigned int m = v_signmask(v_reinterpret_as_s8(max0));

                            for( k = 0; m > 0 && k < 16; k++, m >>= 1 )
                            {
                                if( m & 1 )
                                {
                                    cornerpos[ncorners++] = j+k;
                                    if(nonmax_suppression)
                                    {
                                        short d[25];
                                        for (int _k = 0; _k < 25; _k++)
                                            d[_k] = (short)(ptr[k] - ptr[k + pixel[_k]]);

                                        v_int16x8 a0, b0, a1, b1;
                                        a0 = b0 = a1 = b1 = v_load(d + 8);
                                        for(int shift = 0; shift < 8; ++shift)
                                        {
                                            v_int16x8 v_nms = v_load(d + shift);
                                            a0 = v_min(a0, v_nms);
                                            b0 = v_max(b0, v_nms);
                                            v_nms = v_load(d + 9 + shift);
                                            a1 = v_min(a1, v_nms);
                                            b1 = v_max(b1, v_nms);
                                        }
                                        curr[j + k] = (uchar)(v_reduce_max(v_max(v_max(a0, a1), v_setzero_s16() - v_min(b0, b1))) - 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
#endif
            for( ; j < img.cols - 3; j++, ptr++ )
            {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + MAX16;
                for (int i=0 ; i < 16 ; ++i){
                    // printf("pixel[%d]: %d, ptr[pixel[%d]]: %d\n", i, pixel[i], i, ptr[pixel[i]]);
                }
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3] + 1; // cornerpos[-1] is used to store a value
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
               (score > prev[j+1] && score > prev[j-1] &&
                score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.push_back(KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
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