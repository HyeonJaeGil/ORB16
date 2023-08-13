import cv16
import cv2
import numpy as np
from test_utils import *
from test_pyorb import test_detect, test_kpts_conversion


if __name__ == "__main__":
    ## test pyfast16 (16-bit version)
    img16 = cv2.imread("assets/tir.png", -1)
    img8 = normalize_minmax(img16)
    mask16 = np.zeros(img16.shape, dtype=np.uint8)
    fast16 = cv16.FastFeatureDetector16_create(threshold=40)

    test_detect(fast16, img16, mask=mask16)
    kpts = test_detect(fast16, img16, mask=None)
    print("kpts: ", len(kpts))
    cv2_kpts = test_kpts_conversion(kpts)

    ## test cv2.ORB (reference)
    mask8 = np.zeros(img8.shape, dtype=np.uint8)
    fast8 = cv2.FastFeatureDetector_create()

    test_detect(fast8, img8, mask=mask8)
    kpts = test_detect(fast8, img8, mask=None)


    

