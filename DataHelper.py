import os
import cv2
import consts
import numpy as np
from sklearn.model_selection import train_test_split

def FromOneHot(OneHotEncodedArray):
    normal_array = []
    for target in OneHotEncodedArray:
        normal_array.append(list(target).index(max(list(target))))
    return np.array(normal_array)


def make_read_for_input(path):
    img = cv2.resize(cv2.imread(path),consts.shape_single)
    return np.reshape(img,consts.shape_streamed_one)/255

def OneHotEncoding(array):
    OneHotEncodedArray = []
    for img_label in array:
        target = [0] * consts.classes
        target[img_label] = 1
        OneHotEncodedArray.append(target)
    return np.array(OneHotEncodedArray)
    
def getPrediction(l):
    return consts.CATS[list(l).index(max(list(l)))]
