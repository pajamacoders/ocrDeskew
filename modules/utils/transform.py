import numpy as np
import cv2

class ResizeAspectRatio(object):
    def __init__(self, square_size,  mag_ratio=1):
        self.square_size = square_size
        self.mag_ratio = mag_ratio

    def __call__(self, inp):
        img = inp['img']
        if len(img.shape)>2:
            height, width, channel = img.shape
        else:
            height, width = img.shape

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
        self.k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        img = cv2.dilate(img, self.k)
        proc = cv2.resize(img, (target_w, target_h), interpolation =  cv2.INTER_AREA)
        inp['org_height'] = target_h
        inp['org_width'] = target_w

        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)

        if len(img.shape)>2:
            resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
            resized[0:target_h, 0:target_w, :] = proc
        else:
            resized = np.zeros((target_h32, target_w32), dtype=np.float32)
            resized[0:target_h, 0:target_w] = proc

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
    '''
    This class resizes image with considering aspect ratio.
    If image is lager than target size, this class performs crop.
    If image is smaller than target size, this class pads 0.
    '''
    def __init__(self, target_size=512):
        self.target_size=target_size
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
        iter = max(h,w)//self.target_size
        for i in range(1,iter):
            h, w = h//2, w//2
            img=cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            h, w = img.shape

        h, w = img.shape
        ratio = self.target_size / max(h, w)    
        
        target_h, target_w = int(h * ratio), int(w * ratio)
        #resize aspect ration and pad
        rsz_img = cv2.resize(img, (target_w, target_h), interpolation =  cv2.INTER_AREA)

        img = self.pad_right(rsz_img)
        inp['img']=img
        return inp

    def pad_right(self, crop):
        if len(crop.shape)==2:
            h,w = crop.shape
            padded_img = np.zeros((self.target_size, self.target_size),dtype=np.float32)
        elif len(crop.shape)==3:
            h,w,c = crop.shape
            padded_img = np.zeros((self.target_size, self.target_size,c), dtype=np.float32)

        else:
            pass
        padded_img[:h,:w]=crop

        return padded_img

class ResizeV2(object):
    def __init__(self, scale=2):
        assert (scale!=0) and (scale&(scale-1))==0, 'scale must be power of 2'
        self.iter = np.log2(scale).astype(int)
        self.target_h=512
        self.target_w=512
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
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        img = cv2.dilate(img, k)
        for i in range(self.iter):
            h, w = h//2, w//2
            img=cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            h, w = img.shape
        ratio = self.target_h / max(h, w)    
      
        target_h, target_w = int(h * ratio), int(w * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation =  cv2.INTER_AREA)
        # if len(img.shape)==2:
        #     h,w= img.shape
        # elif len(img.shape)==3:
        #     h,w,c = img.shape
        # else:
        #     pass
        
        # if h> self.target_h and w> self.target_w:
        #     i, j = np.random.randint(h-self.target_h), np.random.randint(w-self.target_w)
        #     crop = img[i:i+self.target_h, j:j+self.target_w]
        # elif h> self.target_h and w <= self.target_w:
        #     i = np.random.randint(h-self.target_h)
        #     crop = img[i:i+self.target_h]
        # elif h<= self.target_h and w > self.target_w:
        #     j = np.random.randint(w-self.target_w)
        #     crop = img[:,j:j+self.target_w]
        # else:
        #     crop = img

        img = self.pad_right(proc)
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

class ResizeBySize(object):
    def __init__(self, size=(40,40)):
        self.height = size[0]
        self.width = size[1]
    
    def __call__(self, inp):
        img = inp['img']
        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
        inp['img']=img
        return inp

class GaussianNoise(object):
    def __init__(self, ratio=0.5, mean=0, std=3):
        self.ratio=ratio
        self.mean=mean
        self.std=std

    def __call__(self, inp):
        if np.random.rand()<self.ratio:
            img = inp['img']
            h,w = img.shape
            gaussian = np.random.normal(self.mean, self.std, (h,w)).astype(np.float32)
            noisy_img = img.astype(np.float32)+gaussian
            cv2.normalize(noisy_img, noisy_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_img = noisy_img.astype(np.uint8)
            inp['img']=noisy_img
        return inp

class Translate(object):
    def __init__(self,ratio=0.5, tx=5, ty=5):
        self.ratio=ratio
        self.tx = tx
        self.ty = ty
    def __call__(self, inp):
        if np.random.rand()<self.ratio:
            img = inp['img']
            h,w = img.shape
            tx = np.random.randint(0,self.tx)
            ty = np.random.randint(0,self.ty)
            M = np.float64([[1,0,tx],[0,1,ty]])
            dst = cv2.warpAffine(img, M, (w,h))
            inp['img']=dst
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
                img[(h_offset+i*line_width):(h_offset+(i+1)*line_width), np.random.randint(w-100):]=0
            inp['img']=img
        return inp
        

class RandomCropAndPad(object):
    def __init__(self, ratio=0.5, size=(768,768)):
        if isinstance(size, str):
            size=eval(size)
        assert len(size)<=2, 'size must be a tuple of int or just a int'
        assert isinstance(size, int) or (isinstance(size[0], int) and isinstance(size[1],int)), 'element must be type of int'
        self.target_h=size[0]
        self.target_w=size[1]
        self.ratio=ratio
    
    def __call__(self, inp):
        img = inp['img']
        if np.random.rand()<0.5:
            if len(img.shape)==2:
                h,w= img.shape
            elif len(img.shape)==3:
                h,w,c = img.shape
            else:
                pass
            
            if h> self.target_h and w> self.target_w:
                i, j = np.random.randint(h-self.target_h), np.random.randint(w-self.target_w)
                crop = img[i:i+self.target_h, j:j+self.target_w]
            elif h> self.target_h and w <= self.target_w:
                i = np.random.randint(h-self.target_h)
                crop = img[i:i+self.target_h]
            elif h<= self.target_h and w > self.target_w:
                j = np.random.randint(w-self.target_w)
                crop = img[:,j:j+self.target_w]
            else:
                crop = img

            img = self.pad_right(crop)
            inp['img']=img
        return inp

    def pad_right(self, crop):
        scale = np.random.randint(128,256) if np.random.rand()<0.5 else 0
        if len(crop.shape)==2:
            h,w = crop.shape
            padded_img = np.ones((self.target_h, self.target_w),dtype=np.float32)*scale
        elif len(crop.shape)==3:
            h,w,c = crop.shape
            padded_img = np.ones((self.target_h, self.target_w,c), dtype=np.float32)*scale

        else:
            pass
        padded_img[:h,:w]=crop

        return padded_img

class GrayScaleAndResize(object):
    def __init__(self, target_size=512):
        self.target_size = target_size
        self.k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    def __call__(self, inp):
        img = inp['img']
        if len(img.shape)==3:
            h,w,c = img.shape
        else: 
            h,w = img.shape
        r = self.target_size/max(h,w)
        target_h, target_w = int(h*r), int(w*r)
        #color image resize
        img = cv2.resize(img, (target_w,target_h),interpolation = cv2.INTER_AREA)
        scale = np.random.randint(128,256) if np.random.rand()<0.5 else 0
        color = np.ones((self.target_size, self.target_size,c), dtype=np.float32)*scale
        color[:target_h, :target_w]=img
        inp['img']=color
        # gray image
        tmp_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tmp = cv2.resize(tmp_img, (target_w,target_h),interpolation = cv2.INTER_AREA)

        gray = np.ones((self.target_size, self.target_size), dtype=np.float32)*scale
        gray[:target_h, :target_w]=tmp
        inp['gray']=gray
        return inp
        
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
            
            img = inp['img']
            if len(img.shape)==2:
                h,w= img.shape
            elif len(img.shape)==3:
                h,w,c = img.shape
            else:
                pass
            deg1 = np.random.uniform(-self.variant, self.variant+0.1)
            matrix = cv2.getRotationMatrix2D((w/2, h/2), deg1, 1)
            dst = cv2.warpAffine(img, matrix, (w, h),borderValue = (0,0,0))

            deg2 = np.random.uniform(-self.variant-deg1, self.variant-deg1)
            matrix = cv2.getRotationMatrix2D((w/2, h/2), deg2, 1)
            border = 0#np.random.randint(128,256)
            dst = cv2.warpAffine(dst, matrix, (w, h),borderValue = (border,border,border) if np.random.rand()<0.5 else (0,0,0))
            deg = deg1+deg2
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
        if 'gray' in inp.keys():
            inp['gray']=inp['gray']/255.0
        return inp

class Shaper(object):
    def __call__(self, inp):
        img = inp['img']
        if len(img.shape) <3:
            img=np.expand_dims(img, -1)
        inp['img'] = img.transpose(2,0,1)

        if 'gray' in inp.keys():
            gray = inp['gray']
            gray=np.expand_dims(gray, -1) 
            inp['gray']=gray.transpose(2,0,1)
        return inp

class RandomDirection(object):
    def __init__(self, ratio):
        self.degrees = [0,90,180,270]
        self.ratio = eval(ratio) if isinstance(ratio, str) else ratio

    def __call__(self, inp):
        if  np.random.rand()<self.ratio:
            img = inp['img']
            h,w= img.shape
            cls = np.random.randint(0,len(self.degrees))
            deg = self.degrees[cls]
            matrix = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
            dst = cv2.warpAffine(img, matrix, (w, h),borderValue = 0)

            noise = np.random.uniform(-3, 3)
            matrix = cv2.getRotationMatrix2D((w/2, h/2), noise, 1)
            border = np.random.randint(0,256)
            dst = cv2.warpAffine(dst, matrix, (w, h),borderValue = border)
            inp['img'] = dst
            inp['noise']=noise
            
        else:
            deg = 0
            cls = 0
      
        inp['cls']=cls
        inp['degree'] = deg
        return inp