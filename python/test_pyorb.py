import cv2
import cv16
import numpy as np
from test_utils import *
    
def pyorb_kpts_to_cv2_kpts(pyorb_kpts):
    cv2_kpts = []
    for pyorb_kpt in pyorb_kpts:
        cv2_kpt = cv2.KeyPoint()
        cv2_kpt.pt = (pyorb_kpt.pt.x, pyorb_kpt.pt.y)
        cv2_kpt.angle = pyorb_kpt.angle
        cv2_kpt.octave = pyorb_kpt.octave
        cv2_kpt.class_id = pyorb_kpt.class_id
        cv2_kpts.append(cv2_kpt)
    return cv2_kpts

def cv2_kpts_to_pyorb_kpts(cv2_kpts):
    pyorb_kpts = []
    for cv2_kpt in cv2_kpts:
        pyorb_kpt = cv16.KeyPoint()
        pyorb_kpt.pt = cv16.Point2f(cv2_kpt.pt[0], cv2_kpt.pt[1])
        pyorb_kpt.angle = cv2_kpt.angle
        pyorb_kpt.octave = cv2_kpt.octave
        pyorb_kpt.class_id = cv2_kpt.class_id
        pyorb_kpts.append(pyorb_kpt)
    return pyorb_kpts

def test_detect_and_compute(orb, img, mask=None):
    cprint("==> detectAndCompute test"+(" with mask" if mask is not None else ""), "yellow")
    kpts, descs = orb.detectAndCompute(img, mask)
    if mask is not None:
        assert len(kpts) == 0, zero_length_error("kpts")
        cprint("==> pass ", "green")
        return
    assert len(kpts) > 0, zero_length_error("kpts")
    assert descs is not None, none_error("descs")
    cprint("==> pass ", "green")
    print("kpts: ", kpts[0].pt, kpts[0].angle, kpts[0].octave, kpts[0].class_id)
    print("descs: ", descs[0])
    return kpts, descs

def test_detect(orb, img, mask=None):
    cprint("==> detect test"+(" with mask" if mask is not None else ""), "yellow")
    kpts = orb.detect(img, mask)
    if mask is not None:
        assert len(kpts) == 0, zero_length_error("kpts")
        cprint("==> pass ", "green")
        return
    assert len(kpts) > 0, zero_length_error("kpts")
    cprint("==> pass ", "green")
    print("kpts: ", kpts[0].pt, kpts[0].angle, kpts[0].octave, kpts[0].class_id)
    return kpts

def test_compute(orb, img, kpts):
    cprint("==> compute test", "yellow")
    kpts_new, descs = orb.compute(img, kpts)
    if isinstance(descs, np.ndarray) and descs.shape[0] == 0:
        descs = None
    for kpt, kpt_new in zip(kpts, kpts_new):
        assert np.sqrt((kpt.pt[0] - kpt_new.pt[0])**2 + (kpt.pt[1] - kpt_new.pt[1])**2) < 1e-5, \
                    neq_error("kpt.pt[0]", "kpt_new.pt[0]")
        assert kpt.angle == kpt_new.angle, neq_error("kpt.angle", "kpt_new.angle")
        assert kpt.octave == kpt_new.octave, neq_error("kpt.octave", "kpt_new.octave")
        assert kpt.class_id == kpt_new.class_id, neq_error("kpt.class_id", "kpt_new.class_id")
    assert len(kpts_new) == len(kpts), neq_error("len(kpts_new)", "len(kpts)")
    if len(kpts) == 0: # descs is nontype
        assert descs is None, notnone_error("descs")
    else:
        assert descs.shape[0] == len(kpts), neq_error("descs.shape[0]", "len(kpts)")
        assert descs.shape[1] == 32, neq_error("descs.shape[1]", "32")
    cprint("==> pass ", "green")

def test_kpts_conversion(pyorb_kpts):
    cprint("==> kpts conversion test", "yellow")
    cv2_kpts = pyorb_kpts_to_cv2_kpts(pyorb_kpts)
    assert len(cv2_kpts) == len(pyorb_kpts), neq_error("len(kpts)", "len(pyorb_kpts)")
    assert isinstance(cv2_kpts[0], cv2.KeyPoint), noninstance_error("cv2_kpts[0]", "cv2.KeyPoint")

    pyorb_kpts = cv2_kpts_to_pyorb_kpts(cv2_kpts)
    assert len(cv2_kpts) == len(pyorb_kpts), neq_error("len(kpts)", "len(pyorb_kpts)")
    assert isinstance(pyorb_kpts[0], cv16.KeyPoint), noninstance_error("pyorb_kpts[0]", "cv16.KeyPoint")
    
    cv2_kpts = pyorb_kpts_to_cv2_kpts(pyorb_kpts)
    assert len(cv2_kpts) == len(pyorb_kpts), neq_error("len(kpts)", "len(pyorb_kpts)")
    assert isinstance(cv2_kpts[0], cv2.KeyPoint), noninstance_error("cv2_kpts[0]", "cv2.KeyPoint")

    cprint("==> pass ", "green")
    return cv2_kpts


if __name__ == "__main__":
    ## test cv16 (16-bit version)
    img16 = cv2.imread("assets/tir.png", -1)
    img8 = normalize_minmax(img16)
    mask16 = np.zeros(img16.shape, dtype=np.uint8)
    orb16 = cv16.ORB16_create(nfeatures=2000, fastThreshold=40)

    test_detect_and_compute(orb16, img16, mask=mask16)
    kpts, descs = test_detect_and_compute(orb16, img16, mask=None)

    test_detect(orb16, img16, mask=mask16)
    kpts = test_detect(orb16, img16, mask=None)
    cv2_kpts = test_kpts_conversion(kpts)
    test_compute(orb16, img16, kpts[0:10])
    test_compute(orb16, img16, [])


    ## test cv2.ORB (reference)
    mask8 = np.zeros(img8.shape, dtype=np.uint8)
    orb8 = cv2.ORB_create(nfeatures=2000)

    test_detect_and_compute(orb8, img8, mask=mask8)
    kpts, descs = test_detect_and_compute(orb8, img8, mask=None)

    test_detect(orb8, img8, mask=mask8)
    kpts = test_detect(orb8, img8, mask=None)
    test_compute(orb8, img8, kpts)
    test_compute(orb8, img8, [])

    # optional: draw keypoints
    def drawKeypoints(img, kpts):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.drawKeypoints(img, kpts, 1, color=(0, 255, 0), flags=0)
        return img

    def getText(bits, feature_num=None, fast_thr=None):
        text = str(bits) + "-bit"
        if feature_num is not None:
            text += ", " + str(feature_num) + " features"
        if fast_thr is not None:
            text += ", fastThr=" + str(fast_thr)
        return text

    orb16_20 = cv16.ORB16_create(nfeatures=2000, fastThreshold=20)
    orb16_40 = cv16.ORB16_create(nfeatures=2000, fastThreshold=40)
    orb8_20 = cv2.ORB_create(nfeatures=2000, fastThreshold=20)

    kpts16_20 = orb16_20.detect(img16, None)
    cv2_kpts16_20 = pyorb_kpts_to_cv2_kpts(kpts16_20)
    kpts16_20_num = len(kpts16_20)
    img16_20_kpts = drawKeypoints(img8, cv2_kpts16_20)
    cv2.putText(img16_20_kpts, getText(16, kpts16_20_num, 20), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    kpts16_40 = orb16_40.detect(img16, None)
    cv2_kpts16_40 = pyorb_kpts_to_cv2_kpts(kpts16_40)
    kpts16_40_num = len(kpts16_40)
    img16_40_kpts = drawKeypoints(img8, cv2_kpts16_40)
    cv2.putText(img16_40_kpts, getText(16, kpts16_40_num, 40), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    kpts8 = orb8.detect(img8, None)
    kpts8_num = len(kpts8)
    img8_kpts = drawKeypoints(img8, kpts8)
    cv2.putText(img8_kpts, getText(8, kpts8_num), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    concat = cv2.hconcat([img16_20_kpts, img16_40_kpts, img8_kpts])
    cv2.imshow("keypoints", concat)
    cv2.imwrite("assets/keypoints.png", concat)

    cv2.waitKey(0)
    

