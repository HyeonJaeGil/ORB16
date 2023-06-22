import cv2
import pyORB16
import numpy as np
from termcolor import colored, cprint

def zero_length_error(name):
    return colored("len({}) == 0".format(name), "red")

def nonzero_length_error(name):
    return colored("len({}) > 0".format(name), "red")

def none_error(name):
    return colored("{} is None".format(name), "red")

def notnone_error(name):
    return colored("{} is not None".format(name), "red")

def neq_error(name1, name2):
    return colored("{} != {}".format(name1, name2), "red")

def noninstance_error(name, type):
    return colored("{} is not {}".format(name, type), "red")

def normalize_minmax(image):
    min_value = image.min()
    max_value = image.max()
    image = image.astype(float)
    image = (image - min_value) / (max_value - min_value)
    image = (image * 255).astype("uint8")
    return image
    
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
        pyorb_kpt = pyORB16.KeyPoint()
        pyorb_kpt.pt = pyORB16.Point2f(cv2_kpt.pt[0], cv2_kpt.pt[1])
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
    assert isinstance(pyorb_kpts[0], pyORB16.KeyPoint), noninstance_error("pyorb_kpts[0]", "pyORB16.KeyPoint")
    
    cv2_kpts = pyorb_kpts_to_cv2_kpts(pyorb_kpts)
    assert len(cv2_kpts) == len(pyorb_kpts), neq_error("len(kpts)", "len(pyorb_kpts)")
    assert isinstance(cv2_kpts[0], cv2.KeyPoint), noninstance_error("cv2_kpts[0]", "cv2.KeyPoint")

    cprint("==> pass ", "green")
    return cv2_kpts


if __name__ == "__main__":
    ## test pyORB16 (16-bit version)
    img16 = cv2.imread("assets/tir.png", -1)
    img16_normalized = normalize_minmax(img16)
    mask16 = np.zeros(img16.shape, dtype=np.uint8)
    orb16 = pyORB16.ORB16_create(nfeatures=2000)

    test_detect_and_compute(orb16, img16, mask=mask16)
    kpts, descs = test_detect_and_compute(orb16, img16, mask=None)

    test_detect(orb16, img16, mask=mask16)
    kpts = test_detect(orb16, img16, mask=None)
    cv2_kpts = test_kpts_conversion(kpts)
    test_compute(orb16, img16, kpts[0:10])
    test_compute(orb16, img16, [])

    # optional: draw keypoints
    def drawKeypoints(img, kpts):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.drawKeypoints(img, kpts, 1, color=(0, 255, 0), flags=0)
        return img
    cv2_kpts = pyorb_kpts_to_cv2_kpts(kpts)
    img16_kpts = drawKeypoints(img16_normalized, cv2_kpts)
    cv2.imshow("img16_kpts", img16_kpts)
    cv2.waitKey(0)

    ## test cv2.ORB (reference)
    img8 = cv2.imread("assets/lena.png", 0)
    mask8 = np.zeros(img8.shape, dtype=np.uint8)
    orb8 = cv2.ORB_create(nfeatures=2000)

    test_detect_and_compute(orb8, img8, mask=mask8)
    kpts, descs = test_detect_and_compute(orb8, img8, mask=None)

    test_detect(orb8, img8, mask=mask8)
    kpts = test_detect(orb8, img8, mask=None)
    test_compute(orb8, img8, kpts)
    test_compute(orb8, img8, [])

