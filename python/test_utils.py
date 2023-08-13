import cv2
import cv16
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