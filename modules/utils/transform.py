import torch
import numpy as np
import cv2

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, inp):
        deg = np.random.uniform(-30, 30)
        img = inp['img']
        h,w= img.shape
        matrix = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
        dst = cv2.warpAffine(img, matrix, (w, h),borderValue=255)
        inp['img'] = dst
        inp['degree'] = deg
        return inp

class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    def __call__(self, inp):
        img = inp['img']
        inp['img']= (img-self.mean)/self.std
        return inp


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, inp):
        pass

        