import torch
import numpy as np
import cv2


class Resize(object):
    def __init__(self, scale=4):
        assert (scale!=0) and (scale&(scale-1))==0, 'scale must be power of 2'
        self.iter = np.log2(scale).astype(int)
    
    def __call__(self, inp):
        img = inp['img']
        h,w = img.shape
        inp['org_height']=h
        inp['org_width']=w
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        img = cv2.dilate(img, k)
        for i in range(self.iter):
            h, w = h//2, w//2
            img=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape
        inp['img']=img=cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        return inp

class RandomCrop(object):
    def __init__(self, size=(256,256)):
        if isinstance(size, str):
            size=eval(size)
        assert len(size)<=2, 'size must be a tuple of int or just a int'
        assert isinstance(size, int) or (isinstance(size[0], int) and isinstance(size[1],int)), 'element must be type of int'
        self.half_h=size[0]//2
        self.half_w=size[1]//2
    
    def __call__(self, inp):
        img = inp['img']
        h,w =img.shape
        assert h>self.half_h*2 and w > self.half_w*2, f'image height and width should be larger than {self.height} and {self.width}'

        hmin=self.half_h
        hmax=h-self.half_h
        wmin = self.half_w
        wmax = w-self.half_w

        if hmin==hmax:
            hmax+=1
        if wmin==wmax:
            wmax+=1

        while 1:
            cy, cx = np.random.randint(hmin, hmax), np.random.randint(wmin, wmax)
            tmp = img[cy-self.half_h:cy+self.half_w, cx-self.half_w:cx+self.half_w]
            if tmp.mean()<250:
                break
        inp['img'] = tmp
        inp['crop_cx']=cx
        inp['crop_cy']=cy
        return inp

class RandomRotation(object):
    def __init__(self, ratio, degree, buckets=None):
        self.variant = eval(degree) if isinstance(degree, str) else degree
        self.ratio = eval(ratio) if isinstance(ratio, str) else ratio
        self.buckets = eval(buckets) if isinstance(buckets, str) else buckets

    def __call__(self, inp):
        if  np.random.rand()<self.ratio:
            deg = np.random.uniform(-self.variant, self.variant-0.1)
            img = inp['img']
            h,w= img.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
            dst = cv2.warpAffine(img, matrix, (w, h),borderValue=0)
            inp['img'] = dst
            cls = 1
        else:
            deg = 0
            cls =0
        if self.buckets:
            rad = np.deg2rad(deg)
            range_rad = np.deg2rad(self.variant)
            bucket = int(self.buckets * (rad+range_rad) / (2*range_rad))
            inp['rot_id']=bucket
        inp['cls']=cls
        inp['degree'] = deg
        return inp

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, inp):
        img = inp['img']
        inp['img']= (img-self.mean)/self.std
        inp['mean']=self.mean
        inp['std']=self.std
        return inp


class Shaper(object):
    def __call__(self, inp):
        img = inp['img']
        if len(img.shape) <3:
            img=np.expand_dims(img, -1)
        inp['img'] = img.transpose(2,0,1)
        return inp


        