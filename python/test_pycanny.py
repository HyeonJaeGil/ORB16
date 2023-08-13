import cv16
import cv2
import numpy as np
from test_utils import *


if __name__ == "__main__":
    ## test pyfast16 (16-bit version)
    img16 = cv2.imread("assets/tir.png", -1)
    img8 = normalize_minmax(img16)

    gradient_x = cv2.Sobel(img16, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img16, cv2.CV_32F, 0, 1, ksize=3)

    edge16 = cv16.Canny16(img16, 100, 200)
    edge16_gradient = cv16.Canny16(gradient_x, gradient_y, 100, 200)
    edge8 = cv16.Canny16(img8, 100, 200)
    edge8_ref = cv2.Canny(img8, 100, 200)

    cv2.imshow("edge16", edge16)
    cv2.imshow("edge16_gradient", edge16_gradient)
    cv2.imshow("edge8", edge8)
    cv2.imshow("edge8_ref", edge8_ref)
    cv2.waitKey(0)

    