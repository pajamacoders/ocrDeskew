import numpy as np
import cv2

class ResizeAspectRatio(object):
    def __init__(self, square_size,  mag_ratio=1):
        self.square_size = square_size
        self.mag_ratio = mag_ratio

    def __call__(self, inp):
        img = inp['img']
        height, width, channel = img.shape

        # magnify image size
        target_size = self.mag_ratio * max(height, width)

        # set original image size
        if target_size > self.square_size:
            target_size = self.square_size
        
        ratio = target_size / max(height, width)    
        '''
        ratio = width/height
        if ratio<1.0:
            ratio = calculate_ratio(width,height)
            img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.ANTIALIAS)
        else:
            img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.ANTIALIAS)
        '''
        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation = cv2.INTER_LINEAR)


        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w/2), int(target_h/2))
        inp['img'] = resized
        inp['size_heatmap']=size_heatmap
        inp['resize_ratio'] = ratio
        return inp

    def calculate_ratio(self, width, height):
        '''
        Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
        '''
        ratio = width/height
        if ratio<1.0:
            ratio = 1./ratio
        return ratio

class Resize(object):
    def __init__(self, scale=4):
        assert (scale!=0) and (scale&(scale-1))==0, 'scale must be power of 2'
        self.iter = np.log2(scale).astype(int)
    
    def __call__(self, inp):
        img = inp['img']
        if len(img.shape)==2:
            h,w= img.shape
        elif len(img.shape)==3:
            h,w,c = img.shape
        else:
            pass
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

class RandomLineErasing(object):
    def __init__(self, ratio, min_line_width=5, max_line_width=40):
        self.ratio = ratio
        self.min_line_width = min_line_width
        self.max_line_width = max_line_width

    def __call__(self,inp):
        if np.random.rand()<=self.ratio:
            img = inp['img']
            if len(img.shape)==2:
                h,w= img.shape
            elif len(img.shape)==3:
                h,w,c = img.shape
            else:
                pass
            line_width = np.random.randint(self.min_line_width, self.max_line_width)
            h_offset = np.random.randint(h//2)
            num_line = (h-h_offset)//line_width
            for i in range(num_line):
                img[(h_offset+i*line_width):(h_offset+(i+1)*line_width), np.random.randint(w-100):]=255
            inp['img']=img
        return inp
        



class RandomCropAndPad(object):
    def __init__(self, size=(768,768)):
        if isinstance(size, str):
            size=eval(size)
        assert len(size)<=2, 'size must be a tuple of int or just a int'
        assert isinstance(size, int) or (isinstance(size[0], int) and isinstance(size[1],int)), 'element must be type of int'
        self.target_h=size[0]
        self.target_w=size[1]
    
    def __call__(self, inp):
        img = inp['img']
        if len(img.shape)==2:
            h,w= img.shape
        elif len(img.shape)==3:
            h,w,c = img.shape
        else:
            pass
        
        if h>= self.target_h and w>= self.target_w:
            i, j = np.random.randint(h-self.target_h), np.random.randint(w-self.target_w)
            crop = img[i:i+self.target_h, j:j+self.target_w]
        elif h>= self.target_h and w < self.target_w:
            i = np.random.randint(h-self.target_h)
            crop = img[i:i+self.target_h]
        elif h< self.target_h and w >= self.target_w:
            j = np.random.randint(w-self.target_w)
            crop = img[:,j:j+self.target_w]
        else:
            crop = img

        img = self.pad_right(crop)
        inp['img']=img
        return inp

    def pad_right(self, crop):
        if len(crop.shape)==2:
            h,w = crop.shape
            padded_img = np.zeros((self.target_h, self.target_w),dtype=np.float32)
        elif len(crop.shape)==3:
            h,w,c = crop.shape
            padded_img = np.zeros((self.target_h, self.target_w,c), dtype=np.float32)

        else:
            pass
        padded_img[:h,:w]=crop

        return padded_img

        
class RandomOrientation(object):
    def __init__(self, ratio):
        self.ratio=ratio
    
    def __call__(self, inp):
        img = inp['img']
        flip = 1 if np.random.rand()<self.ratio else 0
        if flip:
            if len(img.shape)==2:
                h,w= img.shape
            elif len(img.shape)==3:
                h,w,c = img.shape
            else:
                pass
            matrix = cv2.getRotationMatrix2D((w/2, h/2), 180, 1)
            img = cv2.warpAffine(img, matrix, (w, h),borderValue=0)
        inp['img']=img
        inp['flip']=flip
        return inp
            

class RandomRotation(object):
    def __init__(self, ratio, degree, buckets=None):
        self.variant = eval(degree) if isinstance(degree, str) else degree
        self.ratio = eval(ratio) if isinstance(ratio, str) else ratio
        self.buckets = eval(buckets) if isinstance(buckets, str) else buckets

    def __call__(self, inp):
        if  np.random.rand()<self.ratio:
            deg = np.random.uniform(-self.variant, self.variant)
            img = inp['img']
            if len(img.shape)==2:
                h,w= img.shape
            elif len(img.shape)==3:
                h,w,c = img.shape
            else:
                pass
            matrix = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
            dst = cv2.warpAffine(img, matrix, (w, h),borderValue=0)
            inp['img'] = dst
            cls = 1
        else:
            deg = 0
            cls = 0
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

class NormalizeMeanStd(object):
    def __init__(self, mean=(0.485,0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)*255.0
        self.std = np.array(std, dtype=np.float32)*255.0

    def __call__(self, inp):

        img = inp['img']
        img = img.astype(np.float32)
        img -= self.mean
        img /= self.std
        inp['img']=img
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

class RandomRotationTest(object):
    def __init__(self, ratio, degree, buckets=None):
        self.variant = eval(degree) if isinstance(degree, str) else degree
        self.ratio = eval(ratio) if isinstance(ratio, str) else ratio
        self.buckets = eval(buckets) if isinstance(buckets, str) else buckets

    def __call__(self, inp):
        if  np.random.rand()<self.ratio:
            deg = np.random.uniform(-self.variant, self.variant)
            #deg = -self.variant if np.random.rand()>0.5 else self.variant #np.random.uniform(-self.variant, self.variant)
            #deg = np.random.uniform(-self.variant, self.variant)
            img = inp['img']
            h,w= img.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
            dst = cv2.warpAffine(img, matrix, (w, h),borderValue=0)
            inp['img'] = dst
            cls = 1
        else:
            deg = 0
            cls = 0
        if self.buckets:
            rad = np.deg2rad(deg)
            range_rad = np.deg2rad(90)
            bucket = int(self.buckets * (rad+range_rad) / (2*range_rad))
            inp['rot_id']=bucket
        inp['cls']=cls
        inp['degree'] = deg
        return inp