import cv2
import pyORB16
import numpy as np

def detect_orb(image, orb=None, return_descriptors=True):
    if orb is None:
        orb = pyORB16.ORB16_create()
    kpts, descs = orb.detectAndCompute(image, None)
    if return_descriptors:
        return kpts, descs
    else:
        return kpts
    
def visualize_orb(image, kpts=None):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if kpts is None:
        kpts = detect_orb(image, return_descriptors=False)
    image = cv2.drawKeypoints(image, kpts, 1, color=(0, 255, 0), flags=0)
    return image

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


def test_detect_and_compute(orb, img, mask=None):
    print("==> detectAndCompute test"+(" with mask" if mask is not None else ""))
    kpts, descs = orb.detectAndCompute(img, mask)
    if mask is not None:
        assert len(kpts) == 0
        print("==> pass ")
        return
    assert len(kpts) > 0
    assert descs is not None
    print("==> pass ")
    print("kpts: ", kpts[0].pt, kpts[0].angle, kpts[0].octave, kpts[0].class_id)
    print("descs: ", descs[0])
    return kpts, descs

def test_detect(orb, img, mask=None):
    print("==> detect test"+(" with mask" if mask is not None else ""))
    kpts = orb.detect(img, mask)
    if mask is not None:
        assert len(kpts) == 0
        print("==> pass ")
        return
    assert len(kpts) > 0
    print("==> pass ")
    print("kpts: ", kpts[0].pt, kpts[0].angle, kpts[0].octave, kpts[0].class_id)
    return kpts

def test_compute(orb, img, kpts):
    print("==> compute test")
    kpts_new, descs = orb.compute(img, kpts)
    for kpt, kpt_new in zip(kpts, kpts_new):
        assert kpt.pt == kpt_new.pt
        assert kpt.angle == kpt_new.angle
        assert kpt.octave == kpt_new.octave
        assert kpt.class_id == kpt_new.class_id
    assert descs is not None
    print("==> pass ")
    print("descs: ", descs[0])

def test_kpts_conversion(pyorb_kpts, img, visualize=True):
    print("==> kpts conversion test")
    kpts = pyorb_kpts_to_cv2_kpts(pyorb_kpts)
    assert len(kpts) == len(pyorb_kpts)
    assert isinstance(kpts[0], cv2.KeyPoint)
    img = cv2.drawKeypoints(img, kpts, 1, color=(0, 255, 0), flags=0)
    if visualize:
        cv2.imshow("img", img)
        cv2.waitKey(0)
    print("==> pass ")


if __name__ == "__main__":
    ## test pyORB16 (16-bit version)
    # img16 = cv2.imread("assets/tir.png", -1)
    # img16_normalized = normalize_minmax(img16)
    # mask16 = np.zeros(img16.shape, dtype=np.uint8)
    # orb16 = pyORB16.ORB16_create(nfeatures=2000)

    # test_detect_and_compute(orb16, img16, mask=mask16)
    # kpts, descs = test_detect_and_compute(orb16, img16, mask=None)
    # test_kpts_conversion(kpts, img16_normalized, visualize=True)

    # test_detect(orb16, img16, mask=mask16)
    # kpts = test_detect(orb16, img16, mask=None)
    # test_kpts_conversion(kpts, img16_normalized, visualize=True)

    ## test cv2.ORB (reference)
    img8 = cv2.imread("assets/lena.png", 0)
    mask8 = np.zeros(img8.shape, dtype=np.uint8)
    orb8 = cv2.ORB_create(nfeatures=2000)

    test_detect_and_compute(orb8, img8, mask=mask8)
    kpts, descs = test_detect_and_compute(orb8, img8, mask=None)

    test_detect(orb8, img8, mask=mask8)
    kpts = test_detect(orb8, img8, mask=None)
    test_compute(orb8, img8, kpts)